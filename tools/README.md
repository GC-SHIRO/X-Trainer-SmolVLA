# X-trainer 环境安装

本目录提供 Ubuntu 24.04 上的 X-trainer SmolVLA 一键环境安装脚本。默认安装同一个环境中完成数据读取、
SmolVLA 全量训练、LoRA 训练、策略服务、Mock 链路和真机部署所需的 Python 依赖。

## 安装前要求

默认环境基线：


| 项目     | 要求                                                |
| ---------- | ----------------------------------------------------- |
| 操作系统 | Ubuntu 24.04 LTS x86_64                             |
| 环境管理 | Conda（Miniconda 或 Anaconda）                      |
| Python   | 3.12，由脚本创建                                    |
| GPU 训练 | NVIDIA GPU，驱动`>= 570.26`，并且 `nvidia-smi` 可用 |
| PyTorch  | 2.8.0 + CUDA 12.8 wheel                             |
| 网络     | 能访问 Conda、PyPI 和 PyTorch wheel 源              |

默认 CUDA 安装只需要兼容的 NVIDIA 驱动，不要求预先安装系统 CUDA Toolkit。脚本不会安装或升级显卡驱动。
只运行 Mock Policy 或进行无 GPU 检查时，可以使用 `--cpu-only`。

## 一键安装

在仓库根目录执行：

```bash
bash tools/install_xtrainer_env.sh
conda activate xtrainer-smolvla
```

安装脚本默认使用国内镜像：Ubuntu、Conda 和 PyPI 使用清华镜像，PyTorch wheel 使用阿里云镜像。换源只对本次
脚本执行有效，不会永久修改系统 apt、Conda 或 pip 配置。需要使用官方源时执行：

```bash
bash tools/install_xtrainer_env.sh --source official
```

脚本默认会：

1. 检查 Ubuntu 24.04 x86_64、Conda 和 NVIDIA 驱动。
2. 通过 `apt-get` 安装编译工具、FFmpeg、Git、USB 和 udev 运行库。
3. 创建或复用 `xtrainer-smolvla` Conda 环境。
4. 安装 Python 3.12、PyTorch 2.8.0 和 TorchCodec 0.6.0。
5. 从当前仓库安装 `training`、`smolvla`、`peft`、`feetech` 和 `intelrealsense` extras。
6. 安装 ModelScope 下载依赖。
7. 验证训练、WebSocket、Feetech 和 RealSense 的关键导入。

如果环境已经存在，脚本会复用并补齐依赖。需要完全重建时使用：

```bash
bash tools/install_xtrainer_env.sh --recreate
```

CPU-only 环境：

```bash
bash tools/install_xtrainer_env.sh --cpu-only
```

系统包已经安装，或当前用户不能使用 `sudo` 时：

```bash
bash tools/install_xtrainer_env.sh --skip-system-packages
```

自定义环境名：

```bash
bash tools/install_xtrainer_env.sh --env-name xtrainer-dev
conda activate xtrainer-dev
```

## 脚本明确不做的事情

- 不下载 SmolVLA 基础模型、训练 checkpoint 或数据集。
- 不安装或升级 NVIDIA 驱动。
- 不修改 `/dev/ttyUSB*` 串口权限或 RealSense udev 规则。
- 不连接 Dobot、Feetech 或 RealSense，也不会下发机械臂动作。

真机部署前仍需根据现场设备配置串口用户组、RealSense USB 权限、Dobot IP 和防火墙规则。

## 下载 SmolVLA 模型权重

完成环境安装后，从 Hugging Face 或 ModelScope 中选择一个下载入口即可。两个脚本默认都把基础模型保存到
`models/smolvla_base`，不需要重复下载。

Hugging Face 版本：

```bash
bash tools/download_smolvla_weights_hf.sh
```

如需使用 Hugging Face 镜像站，可以显式指定 endpoint：

```bash
bash tools/download_smolvla_weights_hf.sh --endpoint https://hf-mirror.com
```

ModelScope 版本：

```bash
bash tools/download_smolvla_weights_modelscope.sh
```

两个版本默认下载 `lerobot/smolvla_base`。已有文件会被复用，下载中断后可以再次执行相同命令。

Hugging Face 自定义模型、保存位置或固定 revision：

```bash
bash tools/download_smolvla_weights_hf.sh \
  --repo-id lerobot/smolvla_base \
  --output-dir /data/models/smolvla_base \
  --revision main
```

ModelScope 使用 `--model-id` 传入模型标识，其他参数保持一致。如果模型需要登录权限，Hugging Face 使用
`hf auth login` 或 `HF_TOKEN`，ModelScope 使用 `MODELSCOPE_API_TOKEN`。使用其他 Conda 环境时传入
`--env-name`。两个脚本都只下载模型文件，不会启动训练、策略服务或真机程序。
