# X-Trainer 部署 SmolVLA 手册

版本：V1.0<br>
日期：2026-09-20<br>
适用代码：`GC-SHIRO/X-Trainer-SmolVLA` `main`

---

## 1. 文档目标

本文用于在 Dobot X-Trainer 双臂平台上完成 SmolVLA 的数据采集、数据转换、全量微调和真机部署。

完整流程为：硬件检查 -> 遥操作采集 -> LeRobot Dataset v2.1 -> 数据校验 -> SmolVLA 全量微调 -> 策略服务 -> 真机执行。

## 2. 总体架构

### 2.1 端到端流程

```text
X-Trainer 硬件
    -> follower / leader / gripper / RealSense 检查
    -> raw episode 采集
    -> LeRobot Dataset v2.1
    -> 数据校验和相机方向校正
    -> SmolVLA 全量微调
    -> checkpoint
    -> policy server
    -> X-Trainer real client
```

### 2.2 项目分工

| 模块 | 位置 | 作用 |
| --- | --- | --- |
| 采集和机器人控制 | `dobot_xtrainer` | 连接 Dobot、leader、夹爪和相机，保存 raw episode。 |
| 数据转换 | `scripts/xtrainer/convert_raw_to_lerobot_2_1.py` | 将 raw episode 转换为 LeRobot Dataset v2.1。 |
| 相机方向校正 | `tools/transform_xtrainer_dataset_images.py` | 生成相机方向校正后的数据集副本。 |
| 数据校验 | `scripts/xtrainer/validate_dataset_v21.py` | 检查字段、维度、统计量、视频和 episode。 |
| 训练 | `scripts/xtrainer/train_smolvla.sh` | 使用 `configs/xtrainer/train_smolvla.yaml` 进行全量微调。 |
| 策略服务 | `scripts/xtrainer/serve_policy.py` | 加载 checkpoint，通过 WebSocket 提供动作。 |
| 真机客户端 | `scripts/xtrainer/run_real.py` | 读取状态和图像，获取策略动作并下发到硬件。 |

### 2.3 关键数据契约

X-Trainer 的真实状态和动作均为 14 维：

| 索引 | 内容 |
| ---: | --- |
| `0-5` | 左臂关节 1-6，单位为弧度。 |
| `6` | 左夹爪，归一化到 `[0,1]`。 |
| `7-12` | 右臂关节 1-6，单位为弧度。 |
| `13` | 右夹爪，归一化到 `[0,1]`。 |

数据集字段固定为：

| 字段 | 形状 | 含义 |
| --- | ---: | --- |
| `observation.state` | `(14,)` | 双臂关节和双夹爪状态。 |
| `action` | `(14,)` | 与状态使用相同顺序的绝对目标动作。 |
| `observation.images.top` | image/video | 顶部相机。 |
| `observation.images.left_wrist` | image/video | 左腕相机。 |
| `observation.images.right_wrist` | image/video | 右腕相机。 |
| `task` | string | 当前 episode 的任务描述。 |

训练配置将 14 维状态和动作补零到 SmolVLA 的 32 维内部上限。训练损失只使用真实动作维度，推理输出会裁回 14 维。不要在数据转换、训练或部署的任一环节改变动作顺序。

## 3. 硬件和软件前置条件

### 3.1 硬件组成

| 硬件 | 数量 | 用途 |
| --- | ---: | --- |
| Dobot follower 机械臂 | 2 | 执行动作。 |
| leader 主手 | 2 | 输入遥操作动作。 |
| Feetech / X-Trainer 夹爪 | 2 | 控制左右夹爪。 |
| Intel RealSense | 3 | 采集顶部、左腕和右腕图像。 |
| GPU 训练机 | 1 | 训练和运行策略服务。 |
| 机器人控制机 | 1 | 连接硬件、采集数据和执行动作。 |

策略服务和机器人控制程序可以运行在同一台 Ubuntu 机器上，也可以分别运行在可信局域网中的两台机器上。

### 3.2 网络和设备约定

默认机械臂地址为：

```text
左臂 follower: 192.168.5.1
右臂 follower: 192.168.5.2
```

部署前检查网络和设备节点：

```bash
ping 192.168.5.1
ping 192.168.5.2
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
```

实际串口和相机序列号以现场配置为准。USB 设备名可能在重新插拔后变化，不能只按枚举顺序判断左右设备。

### 3.3 系统建议

一键安装脚本面向 Ubuntu x86_64，已在 Ubuntu 24.04 LTS 上验证。默认环境使用 Python 3.12、PyTorch 2.8.0 和 CUDA 12.8 wheel；GPU 模式要求 NVIDIA 驱动不低于 `570.26`，不要求预先安装系统 CUDA Toolkit。

## 4. 环境部署

在仓库根目录执行：

```bash
bash tools/install_xtrainer_env.sh
conda activate xtrainer-smolvla
```

安装脚本默认使用国内镜像，但不会永久修改系统软件源。需要使用官方源时执行：

```bash
bash tools/install_xtrainer_env.sh --source official
```

只做数据转换或无 GPU 检查时，可以安装 CPU 环境：

```bash
bash tools/install_xtrainer_env.sh --cpu-only
conda activate xtrainer-smolvla
```

脚本不安装或升级 NVIDIA 驱动，不下载模型和数据集，也不修改串口或 USB 权限。其他安装选项见 [`tools/README.md`](tools/README.md)。

## 5. X-Trainer 硬件配置

### 5.1 配置内容

需要配置左右 Dobot IP、leader 串口、夹爪串口、夹爪 ID 和三台 RealSense 序列号。不要把密码或现场私有设备信息提交到公共仓库。

状态和动作排列必须保持：

```text
left_j1 ... left_j6, left_gripper,
right_j1 ... right_j6, right_gripper
```

### 5.2 配置文件位置

采集侧沿用 `dobot_xtrainer/scripts/dobot_config/dobot_settings.ini`。该文件包含相机序列号、leader 串口、夹爪串口、关节 ID、offset、方向和初始姿态。

本仓库的真机部署默认值位于 `configs/xtrainer/deploy.yaml`，也可以通过 `run_real.py` 的命令行参数覆盖。现场参数必须以实际硬件为准。

### 5.3 自动扫描串口

在 `dobot_xtrainer` 仓库中执行：

```bash
cd /path/to/workspace/dobot_xtrainer
python scripts/1_find_port.py
```

脚本扫描 `/dev/ttyACM*` 和 `/dev/ttyUSB*`，识别左右 leader 与夹爪并写回配置。设备重插后应重新扫描并核对映射。

### 5.4 标定 leader offset

```bash
python scripts/2_get_offset.py
```

标定前将 leader 放到约定的初始姿态，并确认左右串口、`joint_ids`、`append_id` 和波特率正确。

### 5.5 检查相机和硬件

```bash
ping 192.168.5.1
ping 192.168.5.2
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
python scripts/5_camera_read.py
python scripts/xtrainer/check_real_hardware.py --help
python scripts/xtrainer/check_real_hardware.py --execute
```

执行硬件检查前清空工作区，确保急停可立即触达。确认相机序列号、左右映射和关节方向后再开始采集。

## 6. 遥操作与数据采集

### 6.1 启动 follower server

在终端 1 中执行：

```bash
cd /path/to/workspace/dobot_xtrainer
conda activate xtrainer
python experiments/launch_nodes.py --hostname 127.0.0.1 --robot-port 6001
```

启动前确认左右 Dobot IP 可达、控制盒处于 TCP/IP 控制状态，并且所有安全保护已经正确复位。

### 6.2 启动遥操作和采集程序

在终端 2 中执行：

```bash
cd /path/to/workspace/dobot_xtrainer
conda activate xtrainer
python experiments/run_control.py \
  --hostname 127.0.0.1 \
  --robot-port 6001 \
  --show-img True
```

### 6.3 按钮语义

| 操作 | 作用 |
| --- | --- |
| Button A 短按 | leader lock / unlock。 |
| Button A 长按超过 1 秒 | 对应侧 follower servo start / stop。 |
| Button B 按下 | 开始 / 停止 recording。 |

推荐顺序：启动 follower server，启动 `run_control.py`，短按 A 解锁 leader，长按 A 启动 servo，低速确认跟随方向后按 B 录制。停止录制后再停止 servo，并锁定 leader。

### 6.4 采集前准备

确认 follower 上电、Dobot 网络可达、leader 和夹爪串口可用、三路相机画面正常。正式采集前先用低速动作确认左右臂、夹爪和相机映射。

### 6.5 采集输出结构

```text
collect_data/
└── <episode_id>/
    ├── topImg/<frame_id>.jpg
    ├── leftImg/<frame_id>.jpg
    ├── rightImg/<frame_id>.jpg
    └── observation/<frame_id>.pkl
```

每个 `.pkl` 至少包含 14 维 `joint_positions` 和 14 维 `control`。三路图像与观测文件必须使用相同帧号。

### 6.6 采集要求

1. 每个任务先采集少量短 episode，验证转换和训练链路后再扩大数据量。
2. 每条 episode 从稳定初始场景开始，在任务完成后结束。
3. 相机应覆盖目标物体、末端执行器和完整操作区域。
4. leader 动作保持平滑，避免快速、大幅度移动。
5. 训练和部署使用含义一致的任务描述，例如 `将桌面上的方块放入收纳盒`。

### 6.7 采集后检查

```bash
find /data/xtrainer/collect_data -maxdepth 1 -mindepth 1 -type d | wc -l
find /data/xtrainer/collect_data/<episode_id>/observation -name "*.pkl" | wc -l
find /data/xtrainer/collect_data/<episode_id>/topImg -name "*.jpg" | wc -l
find /data/xtrainer/collect_data/<episode_id>/leftImg -name "*.jpg" | wc -l
find /data/xtrainer/collect_data/<episode_id>/rightImg -name "*.jpg" | wc -l
```

同一 episode 的观测和三路图像帧数不一致时，应先处理 raw 数据，再进行转换。

## 7. Raw 数据转换为 LeRobot 格式

本仓库使用 LeRobot Dataset v2.1，并提供从 X-Trainer raw episode 到训练数据集的转换脚本。

### 7.1 转换命令

```bash
python scripts/xtrainer/convert_raw_to_lerobot_2_1.py \
  --raw-root 数据集对应collect_data的目录 \
  --output-root 输出的数据集的文件夹\
  --task "对应任务的提示词" \
  --fps 30 \
  --use-videos \
  --overwrite-output
```

`--output-root` 必须是独立于 raw 数据的新目录。`--overwrite-output` 会替换已有的非空输出目录，执行前必须确认路径正确。需要在坏帧处立即停止时增加 `--fail-on-bad-frames`。

### 7.2 字段映射

| raw 字段 | LeRobot 字段 |
| --- | --- |
| `joint_positions` | `observation.state` |
| `control` | `action` |
| `topImg` | `observation.images.top` |
| `leftImg` | `observation.images.left_wrist` |
| `rightImg` | `observation.images.right_wrist` |
| `--task` | episode task |

输出至少包含 `meta/info.json`、`meta/stats.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`、Parquet 数据和三路 MP4 视频。

### 7.3 相机转换

如果原始数据的右腕相机方向与部署输入不一致，使用以下工具生成校正后的数据集副本：

```bash
python tools/transform_xtrainer_dataset_images.py \
  --input-root /data/xtrainer/dataset_v21 \
  --output-root /data/xtrainer/dataset_v21_camera_aligned
```

该工具保持顶部和左腕视频不变，将右腕视频垂直翻转后再水平翻转，即旋转 180 度。它不会修改源数据集，只支持包含三路 MP4 视频的 v2.1 数据集。

先验证输入和计划操作而不生成输出：

```bash
python tools/transform_xtrainer_dataset_images.py \
  --input-root /data/xtrainer/dataset_v21 \
  --output-root /data/xtrainer/dataset_v21_camera_aligned \
  --dry-run
```

输出目录已经存在时，只有显式传入 `--overwrite-output` 才会替换它。

注意：训练数据需要手动检查内容，确保左右手的翻转情况一致，有利于模型进行学习

### 7.4 如何理解转换

采集程序生成的是便于机器人实时写入的 raw 数据，训练程序需要的是结构固定的 LeRobot Dataset v2.1。转换脚本负责对齐帧、整理字段、生成元数据并编码视频，不会改变任务本身。

转换前后的 episode 数量通常应保持一致。转换成功只表示数据结构可用，不代表数据质量适合训练；仍需执行第 9 节的数据校验，并抽查图像、状态和动作是否对应。

### 7.5 参数怎么选

- `--raw-root`：采集结果目录，其下应直接包含多个 episode 子目录。
- `--output-root`：新数据集目录。应选择与 raw 目录无父子关系的独立路径，避免 `--overwrite-output` 误删原始数据。
- `--task`：这批 episode 的任务描述；训练和部署应使用含义一致的文本。
- `--fps`：应与实际采集帧率一致，X-Trainer 默认使用 `30`。
- `--use-videos`：生成训练所需的 MP4 视频，正式转换建议启用。
- `--fail-on-bad-frames`：遇到坏帧立即停止，适合正式数据制作和质量检查。
- `--overwrite-output`：替换已有输出目录，仅在确认目标目录可以删除时使用。

## 8. 模型数据配置

SmolVLA 训练读取以下原始数据集字段：

```text
observation.state
observation.images.top
observation.images.left_wrist
observation.images.right_wrist
action
task
```

训练配置位于 `configs/xtrainer/train_smolvla.yaml`。关键配置为：

```yaml
policy:
  type: smolvla
  path: lerobot/smolvla_base
  chunk_size: 50
  n_action_steps: 50
  max_state_dim: 32
  max_action_dim: 32

rename_map:
  observation.images.top: observation.images.camera1
  observation.images.left_wrist: observation.images.camera2
  observation.images.right_wrist: observation.images.camera3
```

`rename_map` 只在数据加载时将三路相机映射为 SmolVLA 使用的 `camera1`、`camera2` 和 `camera3`，不会改写数据集文件。数据集的状态和动作仍为 14 维，模型内部补零到 32 维，推理后再裁回真实动作维度。

## 9. 基础模型与数据校验

使用 Hugging Face 下载 SmolVLA 策略和 SmolVLM2 视觉语言骨干：

```bash
bash tools/download_smolvla_weights_hf.sh
```

需要使用 Hugging Face 镜像时：

```bash
bash tools/download_smolvla_weights_hf.sh --endpoint https://hf-mirror.com
```

也可以使用 ModelScope：

```bash
bash tools/download_smolvla_weights_modelscope.sh
```

默认下载到：

```text
models/smolvla_base
models/smolvlm2_500m_video_instruct
```

已有文件会被复用。训练脚本检测到本地 `config.json` 后，会自动使用这两个本地目录。

训练前对最终使用的数据集执行完整校验：

```bash
python scripts/xtrainer/validate_dataset_v21.py \
  --root /data/xtrainer/dataset_v21_camera_aligned \
  --all-episodes
```

校验器会检查 v2.1 元数据、字段、14 维状态与动作、episode 索引、统计量和视频可读性。只有在校验通过后才开始训练。

## 10. SmolVLA 全量微调

### 10.1 Smoke training

正式训练前，先使用有效的小型数据集完成一次最小训练：

```bash
bash scripts/xtrainer/train_smolvla.sh \
  --dataset-root /data/xtrainer/smoke_v21 \
  --device cuda \
  --batch-size 1 \
  --steps 1 \
  --output-dir outputs/train/xtrainer_smolvla_smoke
```

Smoke training 会执行一次真实的前向传播、反向传播和参数更新，用于验证环境、模型、视频解码和数据管线，不代表模型已经学会任务。

### 10.2 正式训练

```bash
bash scripts/xtrainer/train_smolvla.sh \
  --dataset-root /data/xtrainer/dataset_v21_camera_aligned \
  --device cuda \
  --batch-size 8 \
  --steps 80000 \
  --output-dir outputs/train/xtrainer_smolvla_full
```

训练脚本默认先抽样校验数据集。显存不足时先减小 `--batch-size`；只有在已经独立完成数据校验时才使用 `--skip-validation`。训练输出中 checkpoint 下的 `pretrained_model` 目录用于策略服务。

## 11. 断点续训

`--resume-checkpoint` 可以指向 checkpoint 的 `pretrained_model` 目录或其中的 `train_config.json`：

```bash
bash scripts/xtrainer/train_smolvla.sh \
  --dataset-root /data/xtrainer/dataset_v21_camera_aligned \
  --resume-checkpoint outputs/train/xtrainer_smolvla_full/checkpoints/last/pretrained_model \
  --device cuda \
  --batch-size 8 \
  --steps 100000 \
  --output-dir outputs/train/xtrainer_smolvla_full
```

断点续训以 checkpoint 中保存的策略配置和 processor 为准，仍可显式覆盖数据集根目录、输出目录、设备、batch size 和总步数。更换数据集或任务描述后，应重新确认训练结果是否仍然可比较。

## 12. 启动策略服务

在 GPU 策略机上启动完成微调的 checkpoint：

```bash
conda activate xtrainer-smolvla
python scripts/xtrainer/serve_policy.py \
  --checkpoint outputs/train/xtrainer_smolvla_full/checkpoints/last/pretrained_model \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000 \
  --actions-per-chunk 50
```

未通过命令行指定的参数从 `configs/xtrainer/deploy.yaml` 读取：

| 类型 | 参数 | 默认值或来源 | 作用和注意事项 |
| --- | --- | --- | --- |
| 配置必查 | `--checkpoint` / `--model-path` | `policy.checkpoint` | 要部署的 `pretrained_model` 目录；两个参数名等价。 |
| 可选 | `--config` | `configs/xtrainer/deploy.yaml` | 部署配置文件路径。 |
| 可选 | `--device` | `policy.device`，当前为 `cuda` | 模型加载和推理设备。 |
| 可选 | `--host` | `network.host`，当前为 `0.0.0.0` | 服务监听地址；`0.0.0.0` 表示监听本机所有网络接口。 |
| 可选 | `--port` | `network.port`，当前为 `8000` | WebSocket 服务端口，必须与客户端一致。 |
| 可选 | `--actions-per-chunk` / `--use-length` | `policy.actions_per_chunk`，当前为 `50` | 每次推理最多返回的动作步数；两个参数名等价，设置超过 checkpoint 的 `chunk_size` 也不会产生更多动作。 |
| 可选 | `--log-actions` | 关闭 | 将返回动作块和输入写入 JSONL，仅建议短时调试。 |
| 可选 | `--action-log-path` | `outputs/xtrainer/action_logs/actions_<UTC>.jsonl` | 自定义服务端动作日志路径，仅在启用 `--log-actions` 时生效。 |
| 可选 | `--no-warmup` | 关闭 | 跳过服务启动时的首次 warmup 推理；正常部署建议保留 warmup。 |

服务协议没有认证和 TLS，只能运行在可信局域网中，不要将端口直接暴露到公网。服务启动后保持终端运行。

## 13. 启动真机任务

首次运行使用较短动作 horizon 和较少控制步数：

```bash
conda activate xtrainer-smolvla
python scripts/xtrainer/run_real.py \
  --host <策略机局域网IP> \
  --port 8000 \
  --task "将桌面上的方块放入收纳盒" \
  --left-robot-ip 192.168.5.1 \
  --right-robot-ip 192.168.5.2 \
  --action-horizon 5 \
  --control-hz 30 \
  --max-steps 100 \
  --execute
```

`--host` 填写策略机的局域网 IP；只有服务端和客户端运行在同一台机器上时才使用 `127.0.0.1`。`run_real.py` 的主要参数如下：

| 类型 | 参数 | 默认值 | 作用和注意事项 |
| --- | --- | --- | --- |
| 必须传入 | `--host` | 无 | SmolVLA 策略服务的 IP 地址或主机名。 |
| 必须传入 | `--execute` | 关闭 | 显式允许连接、使能并移动真机；不传时程序拒绝执行。 |
| 真机必查 | `--task` | `pick up the object` | 发送给模型的任务文本，应与训练数据含义一致。 |
| 真机必查 | `--left-robot-ip` / `--right-robot-ip` | `192.168.5.1` / `192.168.5.2` | 左右 Dobot 地址，不能接反。 |
| 真机必查 | `--left-gripper-port` / `--right-gripper-port` | `/dev/ttyUSB1` / `/dev/ttyUSB0` | 左右夹爪串口，USB 重插后必须重新核对。 |
| 真机必查 | `--camera-top-serial` / `--camera-left-wrist-serial` / `--camera-right-wrist-serial` | `409122273405` / `412622272997` / `412622271417` | 顶部、左腕和右腕 RealSense 序列号。 |
| 可选 | `--port` | `8000` | 策略服务端口，必须与服务端一致。 |
| 可选 | `--action-horizon` | `50` | 每个返回动作块最多执行的步数；首次上机建议使用较小值。 |
| 可选 | `--control-hz` | `30` | 动作下发频率。 |
| 可选 | `--max-steps` | `1000` | 整次任务最多执行的控制步数，不是动作块数量。 |
| 可选 | `--async-observation-mode` | `latest` | `latest` 持续提交最新观测；`legacy` 回退为单请求模式。 |
| 可选 | `--background-observation` | 开启 | 在 `latest` 模式使用单后台线程采集观测；`--no-background-observation` 关闭。 |
| 可选 | `--observation-hz` | `10` | 最新观测的最大发送频率，动作仍按 `--control-hz` 下发。 |
| 可选 | `--prefetch-threshold` | `0.7` | 剩余动作比例达到阈值时预取下一动作块，范围为 `[0,1]`。 |
| 可选 | `--chunk-blend-steps` | `6` | 换块时用固定步数衰减双臂关节衔接偏差；`0` 关闭。 |
| 可选 | `--chunk-smoothing-strength` | `0.5` | 对块内双臂关节做三点平滑，范围为 `[0,1]`；`0` 关闭。 |
| 可选 | `--max-joint-delta` | `inf`，关闭 | 环境层单步关节变化上限，单位为弧度。 |
| 可选 | `--max-gripper-delta` | `inf`，关闭 | 环境层单步夹爪变化上限。 |
| 可选 | `--max-delta-per-step` | `0`，关闭 | 客户端下发前对全部 14 维动作施加的单步变化上限。 |
| 可选 | `--ramp-step` | `0.01` | 移动到 reset pose 时用于估算插值步数的关节变化量。 |
| 可选 | `--ramp-max-steps` | `100` | reset pose 插值的最大步数。 |
| 可选 | `--camera-warmup-frames` | `10` | 每台相机启动后丢弃的预热帧数。 |
| 可选 | `--observation-similarity-epsilon` | 关闭 | `latest` 模式下 12 个机械臂关节差的 L2 阈值；不比较图像。 |
| 可选 | `--request-timeout` | `10.0` | 策略请求超时时间，单位为秒。 |
| 可选 | `--log-control` | 关闭 | 将状态、模型动作和最终下发动作写入客户端 JSONL。 |
| 可选 | `--control-log-path` | `outputs/xtrainer/control_logs/control_<UTC>.jsonl` | 自定义客户端日志路径，仅在启用 `--log-control` 时生效。 |

真实策略 metadata 包含 14 维 `reset_pose`。客户端在机械臂使能后会先从当前位置平滑移动到该姿态，再开始请求模型动作。首次运行前必须确认 reset pose、左右臂映射、关节单位和夹爪方向适合当前工作台。

`latest` 模式下，客户端在控制循环执行动作的同时提交新观测，服务端只保留尚未开始推理的最新一条。客户端默认在动作块内部做关节平滑，并在切换动作来源时进行 6 步衔接；夹爪保持模型目标。需要回退时使用 `--async-observation-mode legacy`。

首次真机执行前必须完成硬件检查、清空双臂工作区并确保急停可立即触达。确认短流程稳定后，每次只增加一项参数，例如 `--max-steps` 或 `--action-horizon`，便于区分模型、网络和硬件问题。

## 14. 关键文件索引

```text
tools/install_xtrainer_env.sh
tools/download_smolvla_weights_hf.sh
tools/download_smolvla_weights_modelscope.sh
tools/transform_xtrainer_dataset_images.py
scripts/xtrainer/convert_raw_to_lerobot_2_1.py
scripts/xtrainer/validate_dataset_v21.py
scripts/xtrainer/train_smolvla.sh
scripts/xtrainer/serve_policy.py
scripts/xtrainer/run_real.py
configs/xtrainer/train_smolvla.yaml
configs/xtrainer/deploy.yaml
```

更完整的环境安装选项见 [`tools/README.md`](tools/README.md)，SmolVLA 训练和部署的补充说明见 [`docs/XTRAINER_SMOLVLA.md`](docs/XTRAINER_SMOLVLA.md)。
