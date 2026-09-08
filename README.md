# X-Trainer 部署 SmolVLA 手册

版本：V1.0  
日期：2026-09-06
适用代码：`GC-SHIRO/X-Trainer-SmolVLA` `main`

---

## 1. 文档目标

本文用于在 Dobot X-Trainer 双臂平台上完成 SmolVLA 的数据采集、数据转换、全量微调和真机部署。

流程：硬件检查 -> 遥操作采集 -> LeRobot Dataset v2.1 -> 数据校验 -> SmolVLA 全量微调 -> 策略服务 -> 真机执行。

## 2. 总体架构

### 2.1 端到端流程

```text
X-Trainer 硬件
    -> follower / leader / gripper / RealSense 检查
    -> raw episode 采集
    -> LeRobot Dataset v2.1
    -> SmolVLA 全量微调
    -> checkpoint
    -> policy server
    -> X-Trainer real client
```

### 2.2 项目分工

| 模块 | 位置 | 作用 |
|---|---|---|
| 采集和机器人控制 | `dobot_xtrainer` | 连接 Dobot、leader、夹爪和相机，保存 raw episode。 |
| 数据转换 | `scripts/xtrainer/convert_raw_to_lerobot_2_1.py` | 写入 LeRobot Dataset v2.1。 |
| 数据校验 | `scripts/xtrainer/validate_dataset_v21.py` | 检查字段、维度、统计量、视频和 episode。 |
| 训练 | `scripts/xtrainer/train_smolvla.sh` | 使用 `configs/xtrainer/train_smolvla.yaml` 训练。 |
| 策略服务 | `scripts/xtrainer/serve_policy.py` | 加载 checkpoint 并通过 WebSocket 提供动作。 |
| 真机客户端 | `scripts/xtrainer/run_real.py` | 读取状态和图像并下发动作。 |

### 2.3 关键数据契约

| 字段 | 形状 | 含义 |
|---|---:|---|
| `observation.state` | `(14,)` | 左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。 |
| `action` | `(14,)` | 与状态相同顺序的目标动作。 |
| `observation.images.top` | image/video | 顶部相机。 |
| `observation.images.left_wrist` | image/video | 左腕相机。 |
| `observation.images.right_wrist` | image/video | 右腕相机。 |
| `task` | string | 任务描述。 |

动作索引固定为 `0-5` 左臂、`6` 左夹爪、`7-12` 右臂、`13` 右夹爪；关节单位为弧度，夹爪范围为 `[0,1]`。

## 3. 硬件和软件前置条件

### 3.1 硬件组成

| 硬件 | 数量 | 用途 |
|---|---:|---|
| Dobot follower 机械臂 | 2 | 执行动作。 |
| leader 主手 | 2 | 输入遥操作动作。 |
| Feetech / X-Trainer 夹爪 | 2 | 控制左右夹爪。 |
| Intel RealSense | 3 | 顶部、左腕、右腕图像。 |
| GPU 训练机 | 1 | 训练和策略服务。 |
| 机器人控制机 | 1 | 连接硬件、采集和执行。 |

### 3.2 网络和设备约定

```text
左臂 follower: 192.168.5.1
右臂 follower: 192.168.5.2
```

```bash
ping 192.168.5.1
ping 192.168.5.2
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
```

实际串口和相机序列号以现场配置为准，不能只按 USB 枚举顺序判断左右设备。

### 3.3 系统建议

安装脚本按 Ubuntu 24.04 x86_64、Python 3.12、PyTorch 2.8.0 CUDA 12.8 准备。GPU 模式要求 NVIDIA 驱动不低于 `570.26`。

## 4. 环境部署

在仓库根目录执行：

```bash
bash tools/install_xtrainer_env.sh
conda activate xtrainer-smolvla
```

使用官方软件源：

```bash
bash tools/install_xtrainer_env.sh --source official
```

只做数据转换时可使用 CPU 环境：

```bash
bash tools/install_xtrainer_env.sh --cpu-only
conda activate xtrainer-smolvla
```

脚本不安装显卡驱动、不下载模型，也不修改串口权限。

## 5. X-Trainer 硬件配置

### 5.1 配置内容

配置左右 Dobot IP、leader 串口、夹爪串口、夹爪 ID 和三台 RealSense 序列号。不要把密码、私有串口和设备序列号提交到公共仓库。

动作排列必须保持：

```text
left_j1 ... left_j6, left_gripper,
right_j1 ... right_j6, right_gripper
```

### 5.2 配置文件位置

采集侧沿用 `dobot_xtrainer/scripts/dobot_config/dobot_settings.ini`。其中包含相机序列号、leader 串口、夹爪串口、关节 ID、offset、方向和初始姿态。密码和现场设备信息不要提交到公共仓库。

### 5.3 自动扫描串口

```bash
cd /path/to/workspace/dobot_xtrainer
python scripts/1_find_port.py
```

脚本扫描 `/dev/ttyACM*` 和 `/dev/ttyUSB*`，识别左右 leader 与夹爪并写回配置。设备重插后应重新扫描。

### 5.4 标定 leader offset

```bash
python scripts/2_get_offset.py
```

标定前将 leader 放到约定的初始姿态，确认左右串口、`joint_ids`、`append_id` 和波特率正确。

### 5.5 检查相机和硬件

```bash
ping 192.168.5.1
ping 192.168.5.2
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
python scripts/5_camera_read.py
python scripts/xtrainer/check_real_hardware.py --help
python scripts/xtrainer/check_real_hardware.py --execute
```

执行检查前清空工作区并确认急停可触达。确认相机序列号、左右映射和关节方向后再采集。

## 6. 遥操作与数据采集

### 6.1 启动 follower server

在终端 1 执行：

```bash
cd /path/to/workspace/dobot_xtrainer
conda activate xtrainer
python experiments/launch_nodes.py --hostname 127.0.0.1 --robot-port 6001
```

启动前确认左右 Dobot IP 可达、控制盒处于 TCP/IP 控制状态，且没有安全保护未复位。

### 6.2 启动遥操作和采集程序

在终端 2 执行：

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
|---|---|
| Button A 短按 | leader lock / unlock。 |
| Button A 长按超过 1 秒 | 对应侧 follower servo start / stop。 |
| Button B 按下 | 开始 / 停止 recording。 |

推荐顺序：启动 follower server，启动 `run_control.py`，短按 A 解锁 leader，长按 A 启动 servo，确认跟随方向后按 B 录制；停止录制后再停 servo，并锁定 leader。

### 6.4 采集前准备

确认 follower 上电、Dobot 网络可达、leader 和夹爪串口可用、三路相机画面正常。先用低速动作确认左右臂和夹爪方向。

### 6.5 采集输出结构

```text
collect_data/
└── <episode_id>/
    ├── topImg/<frame_id>.jpg
    ├── leftImg/<frame_id>.jpg
    ├── rightImg/<frame_id>.jpg
    └── observation/<frame_id>.pkl
```

每个 pkl 至少包含 `joint_positions` 和 `control`，两者均为 14 维。三路图像和 pkl 必须按同一帧号对应。

### 6.6 采集要求

1. 每个任务先采集少量短 episode，确认转换和训练链路可用。
2. 每条 episode 从稳定初始场景开始，到任务完成后结束。
3. 相机覆盖目标物体、末端执行器和操作区域。
4. leader 动作保持平滑，避免快速大幅移动。
5. 训练和部署使用固定任务描述，例如 `将桌面上的方块放入收纳盒`。

### 6.7 采集后检查

```bash
find /data/xtrainer/collect_data -maxdepth 1 -mindepth 1 -type d | wc -l
find /data/xtrainer/collect_data/<episode_id>/observation -name "*.pkl" | wc -l
find /data/xtrainer/collect_data/<episode_id>/topImg -name "*.jpg" | wc -l
```

帧数不一致时，先处理 raw 数据，再转换。

## 7. Raw 数据转换为 LeRobot 格式

### 7.1 转换命令

```bash
python scripts/xtrainer/convert_raw_to_lerobot_2_1.py \
  --raw-root /data/xtrainer/collect_data \
  --output-root /data/xtrainer/dataset_v21 \
  --task "将桌面上的方块放入收纳盒" \
  --fps 30 \
  --use-videos \
  --overwrite-output
```

`--output-root` 不能指向 raw 目录；需要在坏帧处停止时增加 `--fail-on-bad-frames`。

### 7.2 字段映射

| raw 字段 | LeRobot 字段 |
|---|---|
| `joint_positions` | `observation.state` |
| `control` | `action` |
| `topImg` | `observation.images.top` |
| `leftImg` | `observation.images.left_wrist` |
| `rightImg` | `observation.images.right_wrist` |
| `--task` | episode task |

输出至少包含 `meta/info.json`、`meta/stats.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`、parquet 数据和三路视频。

## 8. 模型数据配置

SmolVLA 读取以下字段，不要改成其他相机名称：

```text
observation.state
observation.images.top
observation.images.left_wrist
observation.images.right_wrist
action
task
```

训练配置为 `configs/xtrainer/train_smolvla.yaml`。状态、动作维度必须为 14，图像视角数量为 3；训练和部署使用相同任务文本及动作顺序。

## 9. 基础模型与数据校验

下载基础策略和视觉语言骨干：

```bash
bash tools/download_smolvla_weights_hf.sh
```

或使用 ModelScope：

```bash
bash tools/download_smolvla_weights_modelscope.sh
```

默认目录为 `models/smolvla_base` 和 `models/smolvlm2_500m_video_instruct`。训练前执行：

```bash
python scripts/xtrainer/validate_dataset_v21.py \
  --root /data/xtrainer/dataset_v21 \
  --all-episodes
```

相机方向不一致时，使用 `tools/transform_xtrainer_dataset_images.py` 生成副本，不覆盖唯一数据源。

## 10. SmolVLA 全量微调

### 10.1 Smoke training

```bash
bash scripts/xtrainer/train_smolvla.sh \
  --dataset-root /data/xtrainer/dataset_v21 \
  --device cuda --batch-size  1 --steps 1 \
  --output-dir outputs/train/xtrainer_smolvla_smoke
```

### 10.2 正式训练

```bash
bash scripts/xtrainer/train_smolvla.sh \
  --dataset-root /data/xtrainer/dataset_v21 \
  --device cuda --batch-size 4 --steps 30000 \
  --output-dir outputs/train/xtrainer_smolvla_full
```

显存不足时先减小 batch size。训练输出的 `pretrained_model` 目录用于策略服务。

## 11. 断点续训

```bash
bash scripts/xtrainer/train_smolvla.sh \
  --dataset-root /data/xtrainer/dataset_v21 \
  --resume-checkpoint outputs/train/xtrainer_smolvla_full/checkpoints/last/pretrained_model \
  --device cuda --batch-size 4 --steps 30000 \
  --output-dir outputs/train/xtrainer_smolvla_full
```

断点目录中的策略配置和 processor 应与数据字段、相机顺序和任务文本一致。

## 12. 启动策略服务

```bash
conda activate xtrainer-smolvla
python scripts/xtrainer/serve_policy.py \
  --checkpoint outputs/train/xtrainer_smolvla_full/checkpoints/last/pretrained_model \
  --device cuda --host 0.0.0.0 --port 8000
```

机器人控制机通过策略机局域网 IP 访问 TCP `8000` 端口，服务终端保持运行。

## 13. 启动真机任务

首次运行使用较小步数和动作增量：

```bash
conda activate xtrainer-smolvla
python scripts/xtrainer/run_real.py \
  --host <策略机IP> --port 8000 \
  --task "将桌面上的方块放入收纳盒" \
  --left-robot-ip 192.168.5.1 --right-robot-ip 192.168.5.2 \
  --action-horizon 5 --control-hz 10 --max-steps 100 \
  --max-joint-delta 0.03 --max-gripper-delta 0.02 --execute
```

`--host` 填策略机局域网 IP；仅在同机运行时使用 `127.0.0.1`。确认 reset pose、左右臂映射和夹爪方向后再增加运行步数，全程保持急停可触达。

## 14. 关键文件索引

```text
tools/install_xtrainer_env.sh
tools/download_smolvla_weights_hf.sh
scripts/xtrainer/convert_raw_to_lerobot_2_1.py
scripts/xtrainer/validate_dataset_v21.py
scripts/xtrainer/train_smolvla.sh
scripts/xtrainer/serve_policy.py
scripts/xtrainer/run_real.py
configs/xtrainer/train_smolvla.yaml
configs/xtrainer/deploy.yaml
```
