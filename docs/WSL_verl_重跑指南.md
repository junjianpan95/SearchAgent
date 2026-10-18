# WSL2 + verl 重跑指南（供更强机器执行）

> 本机（RTX 5060 Ti 16G）因 WSL 未安装 + 单卡显存限制，verl 版 GRPO/GSPO 未在本机跑通。
> 以下步骤供在**多卡 A100/H100 服务器**或**已装 WSL2 的机器**上重跑 verl 版。

## 一、Windows 侧：安装 WSL2（仅本机需要）

管理员 PowerShell 执行：

```powershell
# 1. 安装 WSL2 + Ubuntu（需要重启）
wsl --install -d Ubuntu-22.04

# 2. 重启后验证
wsl --set-default-version 2
wsl -l -v   # 确认 VERSION=2

# 3. 确认 GPU 直通（NVIDIA 驱动已装的话）
wsl nvidia-smi
```

## 二、WSL 内：搭建 verl 环境

```bash
# 进入项目目录
cd /mnt/e/BaiduSyncdisk/1/SearchAgent

# 一键搭建（装 conda + torch + vllm + ray + flash-attn + verl）
bash scripts/setup_wsl.sh
```

## 三、跑 verl GRPO / GSPO

```bash
conda activate verl
cd /mnt/e/BaiduSyncdisk/1/SearchAgent

# GRPO（每题采样 4 条轨迹，两阶段课程奖励）
bash scripts/run_verl_grpo.sh grpo

# GSPO
bash scripts/run_verl_grpo.sh gspo
```

关键参数（`run_verl_grpo.sh` 内已配好，可调）：
- `--actor_rollout_ref.actor.lora_rank 16`：LoRA 训练省显存
- `--actor_rollout_ref.actor.fsdp_config.param_offload true`：参数卸载 CPU
- `--actor_rollout_ref.rollout.n 4`：每题采样 4 条轨迹
- `--trainer.n_gpus_per_node`：多卡改成卡数
- `--algorithm grpo|gspo`

## 四、多卡服务器（A100/H100）全量重跑

若在 Linux 服务器（非 WSL），跳过第一步，直接：

```bash
# 下载模型与数据（ModelScope 国内源）
python scripts/download_models.py --all
python scripts/download_dataset.py

# 全量数据构造（8B 口径：6000 SFT + 1000 RL）
python -m src.data_gen.build_data --config configs/server_full.yaml

# SFT（Qwen3-8B，把 configs/llamafactory_lora.yaml 的 model_name_or_path 改成 8B）
llamafactory-cli train configs/llamafactory_lora.yaml
llamafactory-cli train configs/llamafactory_dora.yaml

# RL（verl 多卡）
bash scripts/run_verl_grpo.sh grpo
bash scripts/run_verl_grpo.sh gspo

# 四方案评测
python scripts/run_compare.py --n 200 --methods base_react lora dora grpo gspo
```

## 五、评测与回填

```bash
# 评测五指标（Answer F1 / Joint F1 / All-Hit / 冗余率 / 工具调用）
python scripts/run_compare.py --n 200 --methods base_react lora dora grpo gspo
# 结果 -> data/compare_report.json
```

跑完把 `data/compare_report.json` 的数字回填到 `docs/项目介绍.md` 的简历条目。
