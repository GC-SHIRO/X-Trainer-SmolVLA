"""Metadata reader for the supported LeRobot Dataset v2.1 subset."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.constants import HF_LEROBOT_HOME

V21_CODEBASE_VERSION = "v2.1"

_DEFAULT_CHUNKS_SIZE = 1_000
_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
_DEFAULT_VIDEO_PATH = "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_REQUIRED_FEATURES = {
    "observation.state",
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "action",
    "task_index",
    "timestamp",
}
_XTRAINER_CAMERA_KEYS = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required v2.1 metadata file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(value).__name__}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required v2.1 metadata file: {path}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Every row in {path} must be a JSON object")
    return rows


class LeRobotDatasetMetadataV21:
    """Read metadata for a local LeRobot v2.1 X-trainer dataset.

    Only the v2.1 layout used by X-trainer is supported: one parquet file and
    one MP4 per camera per episode.  The class never downloads, converts, or
    writes dataset files.
    """

    def __init__(self, repo_id: str, root: str | Path | None = None):
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        if not self.root.is_dir():
            raise FileNotFoundError(f"v2.1 dataset root does not exist: {self.root}")

        self.info = _load_json(self.root / "meta" / "info.json")
        version = self.info.get("codebase_version")
        if version != V21_CODEBASE_VERSION:
            raise ValueError(
                f"Expected LeRobot Dataset {V21_CODEBASE_VERSION}, got {version!r} in "
                f"{self.root / 'meta' / 'info.json'}"
            )

        self.features = self._normalise_features(self.info.get("features"))
        self._validate_xtrainer_features()
        self.stats = self._load_stats()
        self.tasks = self._load_tasks()
        self.episodes = self._load_episodes()
        self._episodes_by_index = {episode["episode_index"]: episode for episode in self.episodes}

    @staticmethod
    def _normalise_features(raw_features: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(raw_features, dict):
            raise ValueError("meta/info.json must contain a 'features' object")
        features: dict[str, dict[str, Any]] = {}
        for key, value in raw_features.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise ValueError("Every feature definition must be a string key and object value")
            feature = dict(value)
            shape = feature.get("shape")
            if isinstance(shape, list):
                feature["shape"] = tuple(shape)
            elif isinstance(shape, tuple):
                feature["shape"] = shape
            else:
                raise ValueError(f"Feature {key!r} must declare a list-shaped 'shape'")
            if not isinstance(feature.get("dtype"), str):
                raise ValueError(f"Feature {key!r} must declare a string 'dtype'")
            features[key] = feature
        return features

    def _validate_xtrainer_features(self) -> None:
        missing = sorted(_REQUIRED_FEATURES - set(self.features))
        if missing:
            raise ValueError(f"Unsupported v2.1 X-trainer dataset; missing required features: {missing}")
        for key in ("observation.state", "action"):
            if self.features[key]["shape"] != (14,):
                raise ValueError(
                    f"Unsupported v2.1 X-trainer dataset; {key!r} must have shape (14,), "
                    f"got {self.features[key]['shape']!r}"
                )
        invalid_cameras = [key for key in _XTRAINER_CAMERA_KEYS if self.features[key]["dtype"] != "video"]
        if invalid_cameras:
            raise ValueError(
                "Unsupported v2.1 X-trainer dataset; the three camera features must use dtype 'video': "
                f"{invalid_cameras}"
            )

    def _load_stats(self) -> dict[str, dict[str, np.ndarray]]:
        raw_stats = _load_json(self.root / "meta" / "stats.json")
        stats: dict[str, dict[str, np.ndarray]] = {}
        for feature, feature_stats in raw_stats.items():
            if not isinstance(feature_stats, dict):
                raise ValueError(f"Statistics for {feature!r} must be an object")
            stats[feature] = {name: np.atleast_1d(np.asarray(value)) for name, value in feature_stats.items()}
        return stats

    def _load_tasks(self) -> dict[int, str]:
        tasks: dict[int, str] = {}
        for row in _load_jsonl(self.root / "meta" / "tasks.jsonl"):
            try:
                task_index, task = int(row["task_index"]), row["task"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Each v2.1 task row must contain integer task_index and string task") from exc
            if not isinstance(task, str) or not task:
                raise ValueError(f"Task {task_index} must be a non-empty string")
            if task_index in tasks:
                raise ValueError(f"Duplicate task_index {task_index} in meta/tasks.jsonl")
            tasks[task_index] = task
        return tasks

    def _load_episodes(self) -> list[dict[str, Any]]:
        episodes = _load_jsonl(self.root / "meta" / "episodes.jsonl")
        normalised: list[dict[str, Any]] = []
        for row in episodes:
            try:
                episode_index, length = int(row["episode_index"]), int(row["length"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Each v2.1 episode row must contain integer episode_index and length") from exc
            if episode_index < 0 or length <= 0:
                raise ValueError(f"Episode {episode_index} must have a non-negative index and positive length")
            normalised.append({**row, "episode_index": episode_index, "length": length})
        normalised.sort(key=lambda episode: episode["episode_index"])
        actual_indices = [episode["episode_index"] for episode in normalised]
        expected_indices = list(range(len(normalised)))
        if actual_indices != expected_indices:
            raise ValueError(
                "Only contiguous v2.1 episode indices starting at zero are supported; "
                f"got {actual_indices}"
            )
        declared_count = self.info.get("total_episodes")
        if declared_count is not None and int(declared_count) != len(normalised):
            raise ValueError(
                f"meta/info.json declares {declared_count} episodes, but meta/episodes.jsonl contains {len(normalised)}"
            )
        declared_frames = self.info.get("total_frames")
        actual_frames = sum(episode["length"] for episode in normalised)
        if declared_frames is not None and int(declared_frames) != actual_frames:
            raise ValueError(
                f"meta/info.json declares {declared_frames} frames, but meta/episodes.jsonl contains {actual_frames}"
            )
        return normalised

    @property
    def fps(self) -> int:
        fps = self.info.get("fps")
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise ValueError(f"meta/info.json must contain a positive fps, got {fps!r}")
        return int(fps)

    @property
    def camera_keys(self) -> list[str]:
        return list(_XTRAINER_CAMERA_KEYS)

    @property
    def video_keys(self) -> list[str]:
        return self.camera_keys

    @property
    def depth_keys(self) -> list[str]:
        # The X-trainer v2.1 contract has three RGB cameras and no depth features.
        return []

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_frames(self) -> int:
        return sum(episode["length"] for episode in self.episodes)

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    def get_episode(self, episode_index: int) -> dict[str, Any]:
        try:
            return self._episodes_by_index[episode_index]
        except KeyError as exc:
            raise IndexError(f"Episode index out of range: {episode_index}") from exc

    @property
    def chunks_size(self) -> int:
        value = self.info.get("chunks_size", _DEFAULT_CHUNKS_SIZE)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"meta/info.json chunks_size must be a positive integer, got {value!r}")
        return value

    def get_data_file_path(self, episode_index: int) -> Path:
        self.get_episode(episode_index)
        template = self.info.get("data_path", _DEFAULT_DATA_PATH)
        if not isinstance(template, str):
            raise ValueError("meta/info.json data_path must be a string")
        return Path(
            template.format(
                chunk_index=episode_index // self.chunks_size,
                episode_chunk=episode_index // self.chunks_size,
                episode_index=episode_index,
            )
        )

    def get_video_file_path(self, episode_index: int, camera_key: str) -> Path:
        self.get_episode(episode_index)
        if camera_key not in self.camera_keys:
            raise KeyError(f"Unsupported X-trainer camera key: {camera_key}")
        template = self.info.get("video_path", _DEFAULT_VIDEO_PATH)
        if not isinstance(template, str):
            raise ValueError("meta/info.json video_path must be a string")
        return Path(
            template.format(
                chunk_index=episode_index // self.chunks_size,
                video_chunk=episode_index // self.chunks_size,
                episode_chunk=episode_index // self.chunks_size,
                episode_index=episode_index,
                video_key=camera_key,
            )
        )
