"""Read-only frame dataset for the supported LeRobot Dataset v2.1 layout."""

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.feature_utils import check_delta_timestamps, get_delta_indices
from lerobot.datasets.video_utils import decode_video_frames

from .metadata import LeRobotDatasetMetadataV21


class LeRobotDatasetV21(Dataset):
    """Read X-trainer v2.1 recordings without converting or mutating them.

    Rows remain inside their source episode: all temporal requests are clamped
    to that episode and the corresponding ``*_is_pad`` mask records the clamp.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str = "pyav",
        return_uint8: bool = True,
    ):
        self.repo_id = repo_id
        self.meta = LeRobotDatasetMetadataV21(repo_id=repo_id, root=root)
        self.root = self.meta.root
        self.image_transforms = image_transforms
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend
        self.return_uint8 = return_uint8
        self.delta_timestamps = delta_timestamps
        self.delta_indices: dict[str, list[int]] | None = None
        if delta_timestamps is not None:
            unknown_keys = sorted(set(delta_timestamps) - set(self.meta.features))
            if unknown_keys:
                raise ValueError(f"delta_timestamps contains unavailable feature(s): {unknown_keys}")
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)

        all_episodes = list(range(self.meta.total_episodes))
        if episodes is None:
            self.episodes = all_episodes
        else:
            invalid = [episode for episode in episodes if episode not in all_episodes]
            if invalid:
                raise IndexError(f"Episode indices out of range: {invalid}")
            self.episodes = list(episodes)
        self._episode_ends: list[int] = []
        total = 0
        for episode_index in self.episodes:
            total += self.meta.get_episode(episode_index)["length"]
            self._episode_ends.append(total)
        self._episode_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.meta.features

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_frames(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return self._episode_ends[-1] if self._episode_ends else 0

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """Replace the transform applied to the three decoded RGB camera tensors."""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None")
        self.image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        """Disable image transforms for subsequently decoded camera frames."""
        self.image_transforms = None

    def _get_episode_and_frame_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Frame index out of range: {index}")
        episode_offset = bisect_right(self._episode_ends, index)
        previous_end = self._episode_ends[episode_offset - 1] if episode_offset else 0
        return self.episodes[episode_offset], index - previous_end

    def _load_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index in self._episode_cache:
            self._episode_cache.move_to_end(episode_index)
            return self._episode_cache[episode_index]

        from pyarrow import parquet as pq  # noqa: PLC0415

        path = self.root / self.meta.get_data_file_path(episode_index)
        if not path.is_file():
            raise FileNotFoundError(f"Missing v2.1 episode parquet file: {path}")
        values = {key: np.asarray(value) for key, value in pq.read_table(path).to_pydict().items()}
        expected_length = self.meta.get_episode(episode_index)["length"]
        actual_length = len(next(iter(values.values()))) if values else 0
        if actual_length != expected_length:
            raise ValueError(
                f"Episode {episode_index} metadata says length={expected_length}, parquet contains {actual_length} rows"
            )
        required_columns = {"observation.state", "action", "task_index", "timestamp"}
        missing = sorted(required_columns - set(values))
        if missing:
            raise ValueError(f"Episode parquet {path} is missing required columns: {missing}")

        self._episode_cache[episode_index] = values
        if len(self._episode_cache) > 2:
            self._episode_cache.popitem(last=False)
        return values

    @staticmethod
    def _tensor(value: Any) -> torch.Tensor:
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32, copy=False)
        if not array.flags.writeable:
            array = array.copy()
        return torch.as_tensor(array)

    def _query_values(
        self, episode_data: dict[str, np.ndarray], key: str, indices: list[int]
    ) -> torch.Tensor:
        if key not in episode_data:
            raise ValueError(f"Episode parquet has no column for requested feature {key!r}")
        return self._tensor(episode_data[key][indices])

    def _query_videos(
        self, episode_index: int, timestamps: np.ndarray, query_indices: dict[str, list[int]]
    ) -> dict[str, torch.Tensor]:
        videos: dict[str, torch.Tensor] = {}
        for camera_key in self.meta.camera_keys:
            indices = query_indices.get(camera_key, [])
            query_timestamps = timestamps[indices].tolist() if indices else [float(timestamps[0])]
            path = self.root / self.meta.get_video_file_path(episode_index, camera_key)
            if not path.is_file():
                raise FileNotFoundError(f"Missing v2.1 camera video: {path}")
            frames = decode_video_frames(
                path,
                query_timestamps,
                self.tolerance_s,
                self.video_backend,
                return_uint8=self.return_uint8,
            )
            videos[camera_key] = frames.squeeze(0) if len(query_timestamps) == 1 else frames
        return videos

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[item_index] for item_index in range(*index.indices(len(self)))]

        episode_index, frame_index = self._get_episode_and_frame_index(index)
        episode = self._load_episode(episode_index)
        episode_length = self.meta.get_episode(episode_index)["length"]
        item: dict[str, Any] = {
            key: self._tensor(values[frame_index])
            for key, values in episode.items()
            if key not in self.meta.camera_keys
        }

        query_indices: dict[str, list[int]] = {}
        if self.delta_indices is not None:
            for key, deltas in self.delta_indices.items():
                indices = [max(0, min(episode_length - 1, frame_index + delta)) for delta in deltas]
                query_indices[key] = indices
                item[f"{key}_is_pad"] = torch.BoolTensor(
                    [frame_index + delta < 0 or frame_index + delta >= episode_length for delta in deltas]
                )
                if key not in self.meta.camera_keys:
                    item[key] = self._query_values(episode, key, indices)

        timestamps = np.asarray(episode["timestamp"], dtype=np.float64)
        if not query_indices:
            query_indices = {camera_key: [frame_index] for camera_key in self.meta.camera_keys}
        else:
            for camera_key in self.meta.camera_keys:
                query_indices.setdefault(camera_key, [frame_index])
        item.update(self._query_videos(episode_index, timestamps, query_indices))

        if self.image_transforms is not None:
            for camera_key in self.meta.camera_keys:
                item[camera_key] = self.image_transforms(item[camera_key])

        task_index = int(np.asarray(episode["task_index"][frame_index]).item())
        try:
            item["task"] = self.meta.tasks[task_index]
        except KeyError as exc:
            raise ValueError(f"Episode {episode_index} references unknown task_index {task_index}") from exc
        return item
