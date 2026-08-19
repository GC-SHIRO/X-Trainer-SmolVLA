# X-trainer SmolVLA 训练指南

本文说明如何使用 X-trainer 双臂机器人采集的数据微调标准 SmolVLA 策略。训练复用现有的
`lerobot-train` 训练循环和只读 LeRobot Dataset v2.1 适配器，不转换、覆盖或改写原始数据集。

## 环境要求

安装 LeRobot 的数据集和 SmolVLA 依赖，并确保数据校验工具与 `lerobot-train` 使用同一个已激活环境：

```bash
pip install -e ".[dataset,smolvla]"
```

如需在单张 CUDA GPU 上训练，请安装与 CUDA 兼容的 PyTorch，并传入 `--device cuda`。

## 数据集目录与契约

传给启动脚本的数据集根目录必须符合以下 LeRobot v2.1 结构：

```text
my_xtrainer_dataset/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.jsonl
│   └── episodes.jsonl
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/
    ├── observation.images.top/episode_000000.mp4
    ├── observation.images.left_wrist/episode_000000.mp4
    └── observation.images.right_wrist/episode_000000.mp4
```

`meta/info.json` 必须声明 `codebase_version: v2.1`、正数 `fps`，并包含以下字段：

- `observation.state`：14 个 `float32` 值。
- `action`：14 个 `float32` 值。
- `observation.images.top`、`observation.images.left_wrist`、
  `observation.images.right_wrist`：视频字段。
- `timestamp`、`episode_index`、`frame_index`、`task_index`。

每个 episode 的 Parquet 文件包含上述非图像字段。`task_index` 会通过 `meta/tasks.jsonl` 解析为传给
SmolVLA 的任务文本。14 维向量顺序固定为：左臂关节 1–6、左夹爪、右臂关节 1–6、右夹爪；夹爪值必须归一化到
`[0, 1]`。

## 单 GPU 全量微调

全量训练启动脚本使用 `configs/xtrainer/train_smolvla.yaml`，其中指定
`dataset.format_version: v2.1` 和 `lerobot/smolvla_base`。在启动 GPU 训练前，脚本会默认抽样校验
数据集及视频。

Linux/macOS Shell：

```bash
bash scripts/xtrainer/train_smolvla.sh \
  --dataset-root /data/xtrainer/my_xtrainer_dataset \
  --device cuda \
  --batch-size 8 \
  --steps 100000 \
  --output-dir outputs/train/xtrainer_smolvla_full
```

使用 `--help` 查看启动脚本帮助。脚本会拒绝缺失或不存在的数据集目录；只有在数据集已校验且明确需要
跳过只读预检时，才使用 `--skip-validation`。

若要执行最小 smoke run，请使用有效的小型数据集并降低 batch size 与 steps：

```bash
bash scripts/xtrainer/train_smolvla.sh \
  --dataset-root /data/xtrainer/smoke \
  --device cuda \
  --batch-size 1 \
  --steps 1 \
  --output-dir outputs/train/xtrainer_smolvla_smoke
```

该命令仍使用正式训练循环：会完成一次前向传播、反向传播和参数更新，并在训练结束时写入 checkpoint。

## 断点续训

传入 checkpoint 的 `pretrained_model` 目录或其中的 `train_config.json`。断点续训时，checkpoint 中保存的
训练配置是权威配置；启动脚本仍会应用显式传入的 dataset root、输出目录、device、batch size 与 steps 覆盖值。

```bash
bash scripts/xtrainer/train_smolvla.sh \
  --dataset-root /data/xtrainer/my_xtrainer_dataset \
  --resume-checkpoint outputs/train/xtrainer_smolvla_full/checkpoints/last/pretrained_model \
  --device cuda
```

第一版仅支持单个本地 v2.1 数据集。streaming、HF Storage Bucket 和多数据集训练会在启动前被拒绝。如需分布式
训练，请直接使用仓库已文档化的 `torchrun` 工作流，并保持
`--dataset.format_version=v2.1` 配置不变。

## LoRA 微调

LoRA 工作流使用 `configs/xtrainer/train_smolvla_lora.yaml` 和
`scripts/xtrainer/train_smolvla_lora.sh`。开始前还需要安装 PEFT 依赖：

```bash
pip install -e ".[peft]"
```

它从 `lerobot/smolvla_base` 开始训练，并固定使用：

```yaml
peft:
  method_type: LORA
  r: 64
  lora_alpha: 64
```

配置不指定 `target_modules`，因此复用 SmolVLA 内置的默认 LoRA 目标模块；EMA 被禁用，且分片并行会被
训练配置拒绝。LoRA 输出目录独立于全量微调输出目录。

```bash
bash scripts/xtrainer/train_smolvla_lora.sh \
  --dataset-root /data/xtrainer/my_xtrainer_dataset \
  --device cuda \
  --batch-size 8 \
  --steps 100000 \
  --output-dir outputs/train/xtrainer_smolvla_lora
```

最小 smoke run 可将 `--batch-size` 和 `--steps` 都设为 `1`。成功 checkpoint 的
`pretrained_model` 目录应包含 `adapter_model.safetensors`、`adapter_config.json`、策略配置和 processor
文件。该 adapter 不是独立模型：部署或重新加载时必须配合其声明的 `lerobot/smolvla_base` base model。
