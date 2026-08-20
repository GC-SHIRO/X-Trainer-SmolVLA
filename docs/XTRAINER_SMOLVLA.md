# X-trainer SmolVLA：训练、Mock 联调与真机部署

本文说明如何使用 X-trainer 双臂机器人采集的数据微调标准 SmolVLA 策略，并完成 Mock Policy 联调和
真机部署。训练复用现有的 `lerobot-train` 训练循环和只读 LeRobot Dataset v2.1 适配器，不转换、覆盖或
改写原始数据集。

完整流程为：准备 v2.1 数据集 → 全量或 LoRA 训练 → Mock Policy 联调机器人端 → 启动真实策略服务 →
小步执行真机任务。本文不包含代码单元测试或模块测试。

## 环境要求

默认运行环境为 Ubuntu 24.04 LTS x86_64，使用 Conda 管理 Python 3.12。仓库提供一键安装脚本，在仓库
根目录执行：

```bash
bash tools/install_xtrainer_env.sh
conda activate xtrainer-smolvla
```

脚本默认安装 PyTorch 2.8.0 CUDA 12.8 wheel，以及训练、LoRA、WebSocket 服务、Feetech 夹爪和
Intel RealSense 依赖。GPU 模式要求 NVIDIA 驱动不低于 `570.26`，但不要求预装系统 CUDA Toolkit。
脚本不会安装显卡驱动、下载模型或数据集，也不会修改串口和 USB 权限。

安装默认使用国内镜像完成 Ubuntu、Conda、PyPI 和 PyTorch 依赖下载，并且不会永久修改系统源配置。如需改用
官方源，执行 `bash tools/install_xtrainer_env.sh --source official`。

只需要运行 Mock Policy 或无 GPU 的机器人端时，可以安装 CPU 环境：

```bash
bash tools/install_xtrainer_env.sh --cpu-only
conda activate xtrainer-smolvla
```

全部选项和环境边界见 [`tools/README.md`](../tools/README.md)。训练和真实 SmolVLA 策略推理建议使用默认
CUDA 环境；CPU 模式不适合实际训练，也不建议用于有实时性要求的模型推理。

## 下载基础模型权重

安装环境后，可以选择 Hugging Face 或 ModelScope。下载脚本会同时下载策略 `lerobot/smolvla_base` 和训练时必需的
视觉语言骨干 `SmolVLM2-500M-Video-Instruct`：

```bash
# Hugging Face
bash tools/download_smolvla_weights_hf.sh

# 或者使用 ModelScope
bash tools/download_smolvla_weights_modelscope.sh
```

默认目录为 `models/smolvla_base` 和 `models/smolvlm2_500m_video_instruct`。离线部署时，将前者传给
`serve_policy.py --checkpoint models/smolvla_base`。自定义模型 ID、保存目录、revision 和 Conda 环境名的方法见
[`tools/README.md`](../tools/README.md)。LoRA adapter 只包含增量参数，因此部署 LoRA 前也必须准备基础模型。

全量训练和 LoRA 启动脚本会在新训练时自动检查两个目录中的 `config.json`。文件存在时，脚本会同时传入本地策略和
本地 VLM 骨干路径，并让 tokenizer 使用同一份本地 VLM，不会访问 Hugging Face。断点续训不会使用这个自动覆盖，
始终以 checkpoint 保存的策略配置为准。

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

基础 SmolVLA checkpoint 使用 `observation.images.camera1`、`camera2`、`camera3` 三个视觉键。X-trainer 的
全量和 LoRA 配置已内置重命名：`top → camera1`、`left_wrist → camera2`、`right_wrist → camera3`。原始数据集
文件和字段不会被修改。

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

`--device` 会覆盖策略的运行设备，可设为 `cuda`、`cuda:0` 或 `cpu`。即使基础模型路径由 YAML 的
`policy.path` 指定，也可以正常传入该参数；策略配置会在加载基础模型时再应用此覆盖值。
X-trainer 的全量与 LoRA 配置默认 `push_to_hub: false`，训练 checkpoint 仅写入本地 `outputs/`，无需提供
Hugging Face `repo_id`。

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
`scripts/xtrainer/train_smolvla_lora.sh`。一键环境脚本已经包含 PEFT 依赖，无需再次安装。

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

## 训练输出如何用于部署

全量训练和 LoRA 训练的输出用途不同：

| 训练方式  | 部署时使用的目录                                  | 启动参数           |
| --------- | ------------------------------------------------- | ------------------ |
| 全量微调  | checkpoint 下的`pretrained_model`               | `--checkpoint`   |
| LoRA 微调 | 含`adapter_config.json` 的 `pretrained_model` | `--lora-adapter` |

全量训练示例目录：

```text
outputs/train/xtrainer_smolvla_full/checkpoints/last/pretrained_model
```

LoRA adapter 示例目录：

```text
outputs/train/xtrainer_smolvla_lora/checkpoints/last/pretrained_model
```

LoRA 加载时，服务会先读取 adapter 配置中记录的基础模型，再把 adapter 叠加到基础模型上。因此 adapter
目录不能被当作完整模型单独使用。

## 部署结构

推荐把策略服务和机器人控制程序分开运行：

```text
GPU 策略机                                         机器人控制机
serve_policy.py  <------ WebSocket / TCP 8000 ----> run_real.py
    SmolVLA                                           Dobot 双臂
                                                      Feetech 双夹爪
                                                      3 台 RealSense
```

如果只有一台 Ubuntu 机器，也可以在两个终端中运行服务端和机器人端，客户端使用 `--host 127.0.0.1`。
服务协议没有认证和 TLS，只能放在可信局域网中；不要把 8000 端口直接暴露到公网。

策略与机器人之间的数据契约固定为：

- 输入：`top`、`left_wrist`、`right_wrist` 三路 RGB 图像，14 维机器人状态和任务文本。
- 输出：14 维绝对目标，顺序为左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。
- 服务一次可以返回最多 50 步动作；机器人端按 `--action-horizon` 决定实际执行多少步后重新请求。
- 真实策略服务会在 metadata 中提供 `reset_pose`；机器人端连接后会先平滑移动到该姿态，再开始策略循环。

## 先运行 Mock Policy 联调

Mock Policy 不加载模型，也不需要 checkpoint。它读取机器人端上传的当前 14 维状态，并返回“保持当前姿态”的
动作块。它适合先确认网络、协议、相机、机械臂和夹爪都能被机器人端正确打开。

注意：Mock 不是纯软件模拟。`run_real.py` 仍然会连接真实 Dobot、Feetech 和 RealSense；添加 `--execute`
后也会向机器人发送保持姿态命令。首次运行前仍须清空工作区、准备急停并由人员看护。

在服务端启动 Mock Policy：

```bash
conda activate xtrainer-smolvla
python scripts/xtrainer/serve_mock_policy.py \
  --host 0.0.0.0 \
  --port 8000 \
  --chunk-size 50
```

在机器人控制机使用较短时长和保守阈值运行：

```bash
conda activate xtrainer-smolvla
python scripts/xtrainer/run_real.py \
  --host 127.0.0.1 \
  --port 8000 \
  --task "保持当前位置，检查部署链路" \
  --action-horizon 5 \
  --control-hz 10 \
  --max-steps 20 \
  --max-joint-delta 0.03 \
  --max-gripper-delta 0.02 \
  --execute
```

将 `192.168.1.100` 替换为 Mock 服务所在机器的局域网 IP。若不传 `--execute`，程序会主动拒绝进入运动流程；
这是防止误操作的安全开关，不是预览模式。

Mock metadata 不包含 `reset_pose`，因此 Mock 联调不会主动把机械臂移动到真实策略的复位姿态。它的正确表现是：
服务持续返回与当前状态相同的目标，机器人没有明显位移，终端没有维度、超时或设备连接错误。

## 真机硬件准备

默认硬件参数与 X-trainer 参考部署保持一致，可以通过 `run_real.py` 参数覆盖：

| 设备           | 默认配置                    |
| -------------- | --------------------------- |
| 左 Dobot       | `192.168.5.1`             |
| 右 Dobot       | `192.168.5.2`             |
| 左夹爪         | `/dev/ttyUSB1`，ID `21` |
| 右夹爪         | `/dev/ttyUSB0`，ID `22` |
| 顶部 RealSense | 序列号`409122273405`      |
| 左腕 RealSense | 序列号`412622272997`      |
| 右腕 RealSense | 序列号`412622271417`      |

部署前逐项确认：

1. 机器人控制机能访问两台 Dobot 的 IP，且 IP 没有接反。
2. 当前用户能访问两个 `/dev/ttyUSB*`；需要时将用户加入 `dialout` 组，重新登录后再运行。
3. 三台 RealSense 的物理安装位置和序列号一致，尤其不能交换左右腕相机。
4. 策略机 TCP 8000 端口可由机器人控制机访问，但只允许可信局域网访问。
5. 双臂周围没有人员、线缆或障碍物，急停可立即触达。
6. 已先完成 Mock 联调，再切换为真实 checkpoint。

串口设备名可能随 USB 插拔顺序变化。如果现场名称不同，显式传入
`--left-gripper-port` 和 `--right-gripper-port`，不要仅凭 `/dev/ttyUSB0`、`/dev/ttyUSB1` 的编号猜测左右。

## 启动真实策略服务

### 全量微调 checkpoint

在有 NVIDIA GPU 的策略机上运行：

```bash
conda activate xtrainer-smolvla
python scripts/xtrainer/serve_policy.py \
  --checkpoint outputs/train/xtrainer_smolvla_full/checkpoints/last/pretrained_model \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000
```

### LoRA adapter

部署 LoRA 时，同时给出基础模型和 adapter。`--checkpoint` 可以是 Hugging Face 模型 ID，也可以是已经下载的
本地基础模型目录：

```bash
conda activate xtrainer-smolvla
python scripts/xtrainer/serve_policy.py \
  --checkpoint lerobot/smolvla_base \
  --lora-adapter outputs/train/xtrainer_smolvla_lora/checkpoints/last/pretrained_model \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000
```

服务启动时会加载策略和 processor，并默认执行一次 warmup。只有在明确不需要 warmup 时才使用
`--no-warmup`。服务正常运行后保持该终端不要退出。

## 启动真机任务

先确认服务端已经启动，再在机器人控制机执行。首次使用真实策略时，建议保持短动作块、低步数和严格的增量限制：

```bash
conda activate xtrainer-smolvla
python scripts/xtrainer/run_real.py \
  --host 192.168.1.100 \
  --port 8000 \
  --task "将桌面上的方块放入收纳盒" \
  --left-arm-ip 192.168.5.1 \
  --right-arm-ip 192.168.5.2 \
  --left-gripper-port /dev/ttyUSB1 \
  --right-gripper-port /dev/ttyUSB0 \
  --top-camera-serial 409122273405 \
  --left-wrist-camera-serial 412622272997 \
  --right-wrist-camera-serial 412622271417 \
  --action-horizon 5 \
  --control-hz 10 \
  --max-steps 100 \
  --max-joint-delta 0.03 \
  --max-gripper-delta 0.02 \
  --max-delta-per-step 0.02 \
  --execute
```

真实策略服务会把 14 维 `reset_pose` 放进 metadata。机器人端会在机械臂使能后，先按照 `--ramp-step` 和
`--ramp-max-steps` 平滑移动到该姿态，然后才请求模型动作。默认复位姿态来自 X-trainer 部署配置；如果该姿态
不适合当前工作台、末端工具或关节限位，应先停止部署并修改服务端配置，不能依赖运行时安全阈值替代人工确认。

确认短流程稳定后，再逐步增加 `--max-steps`、`--action-horizon` 或 `--control-hz`。每次只放宽一项，便于判断
异常来自模型动作、网络延迟还是硬件控制。

## 常见问题

### 服务端能启动，但机器人端连接失败

确认机器人端的 `--host` 使用策略机的局域网 IP，而不是策略机自己的 `127.0.0.1`；同时检查 TCP 8000
端口和防火墙。服务端绑定 `0.0.0.0` 只表示监听所有网卡，它不是机器人端应填写的目标地址。

### 训练启动时提示 `policy: Could not decode ... got {'device': 'cuda'}`

这表示当前代码没有把 `--policy.device` 正确延后到基础模型配置加载阶段，不是数据集校验失败。确认仓库包含
`src/lerobot/configs/parser.py` 的 YAML `policy.path` 二次过滤修复后，保留 `--device cuda` 原样重试即可。

### 提示状态或动作不是 14 维

训练数据、策略 metadata 和真机客户端必须使用同一套 14 维顺序。不要对某一侧单独调整关节或夹爪排列；
应从数据集字段、checkpoint 和部署配置一起检查。

### 夹爪无法连接

先确认串口设备名和 ID，没有权限时配置 `dialout` 用户组并重新登录。左右串口接反会使动作发送给错误夹爪，
因此不应通过反复尝试动作来判断映射。

### RealSense 无法打开

确认三台相机没有被其他程序占用、USB 带宽足够，并核对序列号。顶部、左腕和右腕图像即使分辨率相同也不能
互换，因为训练数据中的语义键是固定的。

### 真实策略连接后机械臂开始复位

这是 `reset_pose` metadata 的预期行为，不代表模型已经开始执行任务。如果实际复位方向或姿态不安全，应立即
急停并检查左右臂映射、关节单位和复位值，不能继续等待策略自行纠正。
