# whichllm

[![PyPI version](https://img.shields.io/pypi/v/whichllm)](https://pypi.org/project/whichllm/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/Andyyyy64/whichllm/actions/workflows/test.yml/badge.svg)](https://github.com/Andyyyy64/whichllm/actions/workflows/test.yml)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-EA4AAA?logo=githubsponsors)](https://github.com/sponsors/Andyyyy64)

<p align="center">
  <a href="https://trendshift.io/repositories/30336" target="_blank"><img src="https://trendshift.io/api/badge/repositories/30336" alt="Andyyyy64%2Fwhichllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

**找到在你的硬件上真正能跑起来的最佳本地大模型。**

自动检测你的 GPU/CPU/内存，并从 HuggingFace 中排名适合你系统的顶级模型。

[English](README.md) | [日本語版](docs/README.ja.md)

## 快速开始

运行推荐命令，无需项目配置。

```bash
uvx whichllm@latest
```

在购买硬件前模拟 GPU。

```bash
uvx whichllm@latest --gpu "RTX 4090"
```

经常使用时建议安装。

```bash
uv tool install whichllm
uv tool upgrade whichllm  # 更新已安装的版本
```

其他安装方式。

```bash
brew install andyyyy64/whichllm/whichllm
pip install whichllm
```

## 想要更稳妥的选择？

默认情况下，whichllm 比较激进。它会推荐在你的机器上看起来能跑的最佳模型，包括部分 RAM 卸载和 VRAM 接近满载的情况。

如果你想要类似 LM Studio 风格的更稳妥推荐，可以使用：

```bash
uvx whichllm@latest --gpu-only --speed usable --vram-headroom 1GB
```

这样只保留完全适配 GPU 显存的模型，过滤掉速度过慢的选项，并为运行时开销预留额外的显存空间。

如果 LM Studio 仍提示模型略大，可以增加预留空间：

```bash
uvx whichllm@latest --gpu-only --speed usable --vram-headroom 1.5GB
```

## 常用场景

安装后直接运行 `whichllm`。一次性使用时，将 `whichllm` 替换为 `uvx whichllm@latest`。

```bash
# 推荐本机最佳模型
whichllm

# 模拟指定 GPU
whichllm --gpu "RTX 4090"

# 手动覆盖 iGPU/统一内存的显存限制
whichllm --vram 8 --ram-bandwidth 68

# 只显示完全适配 GPU 显存的模型
whichllm --gpu-only
whichllm --fit gpu

# 模拟多 GPU 工作站
whichllm --gpu "2x RTX 4090"

# 隐藏虽能运行但速度过慢的模型
whichllm --speed usable
whichllm --speed fast

# 生成可粘贴到 GitHub / Slack / Discord 的输出
whichllm --markdown

# 对比升级候选方案
whichllm upgrade "RTX 4090" "RTX 5090" "H100"

# 查找运行某个模型所需的 GPU
whichllm plan "llama 3 70b"

# 启动模型对话
whichllm run "qwen 2.5 1.5b gguf"

# 输出可直接复制的 Python 代码片段
whichllm snippet "qwen 7b"

# 输出 JSON 格式（适用于脚本）
whichllm --top 1 --json
```

![demo](assets/demo.gif)

## 输出示例

```text
$ whichllm --gpu "RTX 4090"

#1  Qwen/Qwen3.6-27B     27.8B  Q5_K_M   score 92.8    27 t/s
#2  Qwen/Qwen3-32B       32.0B  Q4_K_M   score 83.0    31 t/s
#3  Qwen/Qwen3-30B-A3B   30.0B  Q5_K_M   score 82.7   102 t/s
```

32B 模型**完全适配你的显卡** — 但 whichllm 仍将 27B 排在第一，因为它在真实基准测试中得分更高，且属于更新一代。如果只是一个简单的"能不能装下？"工具，只会推荐更大的那个。这正是 whichllm 的价值所在。（注意第 3 名：一个 MoE 模型达到 102 t/s — 速度按*激活*参数排名，质量按*总*参数评分。）

## 我能跑什么？

实际推荐结果（2026 年 5 月快照 — 你的结果基于 HuggingFace **实时**数据，这不是静态列表）：

| 硬件 | 显存 | 最佳推荐 | 速度 |
|---|---|---|---|
| RTX 5090 | 32 GB | `Qwen3.6-27B` · Q6_K · 评分 94.7 | ~40 t/s |
| RTX 4090 / 3090 | 24 GB | `Qwen3.6-27B` · Q5_K_M · 评分 92.8 | ~27 t/s |
| RTX 4060 | 8 GB | `Qwen3-14B` · Q3_K_M · 评分 71.0 | ~22 t/s |
| Apple M3 Max | 36 GB | `Qwen3.6-27B` · Q5_K_M · 评分 89.4 | ~9 t/s |
| 仅 CPU | — | `gpt-oss-20b` (MoE) · Q4_K_M · 评分 45.2 | ~6 t/s |

`whichllm --gpu "<你的显卡>"` 可在购买前模拟以上任意配置。
默认排名包括完全 GPU 适配、部分卸载和仅 CPU 的候选项（前提是可用）。使用 `--gpu-only` 或 `--fit full-gpu` 可以只显示完全装入 GPU 显存的模型。
默认表格展示显存占用、预估生成速度、适配类型和发布日期。速度按实际可用性以颜色标识：低于 4 tok/s 为红色，4-10 为黄色，10-30 为绿色，30 以上为亮绿色。`~` / `?` 标记估算置信度。

## 为什么选择 whichllm？

让模型塞进显存只是第一步。真正的难点在于知道**在能跑的模型中哪个才是最好的** — 这正是 whichllm 致力于解决的问题。

- **基于实证的排名，而非体积启发式** — 最佳推荐来自多个真实基准测试的合并结果（LiveBench、Artificial Analysis、Aider、多模态/视觉、Chatbot Arena ELO、Open LLM Leaderboard）— 绝不是"刚好能装下的最大模型"。
- **感知时效性** — 过时的排行榜会沿模型族谱被降权，2024 年的老模型无法凭借过期分数超越新一代模型。每个排名下方都会打印基准快照日期，过时的推荐一目了然，而非被默默信任。
- **证据分级与防护** — 每个分数标注为 `direct`（直接匹配）/ `variant`（变体匹配）/ `base`（基础模型）/ `interpolated`（插值）/ `self-reported`（自报），并按置信度折扣。伪造的上传者声明和跨家族继承（小分支借用大基座的分数）会被主动拒绝。
- **架构感知的估算** — VRAM = 权重 + GQA KV 缓存 + 激活值 + 开销；速度受带宽约束，考虑了每种量化效率、后端因子、MoE 激活/总参数拆分，以及统一内存 vs 独立 PCIe 部分卸载建模。
- **一条命令，可脚本化** — `whichllm` 直接输出结果；加 `--json | jq` 可接入管道。没有 TUI，没有需要记忆的快捷键。
- **实时数据** — 模型直接从 HuggingFace API 获取，并提供离线或限速时的备用缓存数据。

## 功能特性

- **自动检测硬件** — NVIDIA、AMD、Intel、Apple Silicon、仅 CPU
- **智能排名** — 根据显存适配、速度和基准测试质量评分
- **一键对话** — `whichllm run` 下载并立即启动对话
- **代码片段** — `whichllm snippet` 输出可直接运行的 Python 代码
- **实时数据** — 从 HuggingFace 直接获取模型（带缓存以提升性能）
- **基准测试感知** — 集成真实评测分数并基于置信度衰减
- **任务配置** — 按通用、编码、视觉或数学场景过滤
- **GPU 模拟** — 测试任意 GPU：`whichllm --gpu "RTX 4090"`
- **多 GPU 模拟** — 重复 `--gpu`、逗号分隔或写 `2x RTX 4090`
- **全显存过滤** — `--gpu-only` / `--fit full-gpu` 隐藏卸载候选
- **速度感知过滤** — `--speed usable|fast` 按阈值隐藏过慢的选项
- **Markdown 输出** — `--markdown` / `-m` 输出可粘贴的 GFM 表格
- **运行时显存预算** — `--vram-headroom` 和 `--ram-budget` 避免极限适配
- **硬件规划** — 反向查询：`whichllm plan "llama 3 70b"`
- **升级规划** — 将当前机器与候选 GPU 进行对比
- **JSON 输出** — 管道友好：`whichllm --json`

## 运行与代码片段

一条命令即可体验任意模型。无需手动安装 — whichllm 通过 `uv` 创建隔离环境，安装依赖，下载模型并启动交互式对话。

![run demo](assets/demo-run.gif)

```bash
# 与模型对话（自动选择最佳 GGUF 变体）
whichllm run "qwen 2.5 1.5b gguf"

# 自动为你的硬件选择最佳模型并对话
whichllm run

# 仅 CPU 模式
whichllm run "phi 3 mini gguf" --cpu-only
```

支持**所有模型格式**：
- **GGUF** — 通过 `llama-cpp-python`（轻量、快速）
- **AWQ / GPTQ** — 通过 `transformers` + `autoawq` / `auto-gptq`
- **FP16 / BF16** — 通过 `transformers`

获取**可复制粘贴的 Python 代码片段**：

```bash
whichllm snippet "qwen 7b"
```

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False,
)

output = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
)
print(output["choices"][0]["message"]["content"])
```

## 用法

```bash
# 自动检测硬件并显示最佳模型
whichllm

# 模拟 GPU（如规划购买）
whichllm --gpu "RTX 4090"
whichllm --gpu "RTX 5090"
# 指定变体
whichllm --gpu "RTX 5060 16"
# 手动覆盖 iGPU/统一内存的显存限制
whichllm --vram 8 --ram-bandwidth 68
# 模拟多块 GPU
whichllm --gpu "2x RTX 4090"
whichllm --gpu "RTX 4090" --gpu "RTX 3090"
whichllm --gpu "RTX 4090, RTX 3090"

# 只显示完全装入 GPU 显存的模型
whichllm --gpu-only
whichllm --fit gpu
whichllm --fit full-gpu

# 避免极限适配和后台内存占用问题
whichllm --vram-headroom 1.5GB
whichllm --ram-budget available
whichllm --ram-budget 8GB

# 仅 CPU 模式
whichllm --cpu-only

# 更多结果 / 过滤
whichllm --top 20
whichllm --details          # 显示下载量元数据而非运行时列
whichllm --speed usable     # 最低 10 tok/s
whichllm --speed fast       # 最低 30 tok/s
whichllm --min-speed 4      # 精确 tok/s 下限
whichllm --markdown         # 可粘贴的 GitHub Flavored Markdown 表格
whichllm --profile coding
whichllm --context-length 64k
whichllm --quant Q4_K_M
whichllm --min-speed 30     # 精确 tok/s 下限
whichllm --evidence base   # 允许 id/基础模型匹配
whichllm --evidence strict # 仅 id 精确匹配（同 --direct）
whichllm --direct

# JSON 输出
whichllm --json

# 强制刷新（忽略缓存）
whichllm --refresh

# 仅显示硬件信息
whichllm hardware

# 规划：运行某个模型需要什么 GPU？
whichllm plan "llama 3 70b"
whichllm plan "Qwen2.5-72B" --quant Q8_0
whichllm plan "mistral 7b" --context-length 32768

# 升级：将当前机器与候选 GPU 进行对比
whichllm upgrade "RTX 4090" "RTX 5090" "H100"
whichllm upgrade "Apple M4 Max" --top 5

# 运行：下载并立即与模型对话
whichllm run "qwen 2.5 1.5b gguf"
whichllm run                       # 自动为硬件选择最佳模型

# 代码片段：输出可直接运行的 Python 代码
whichllm snippet "qwen 7b"
whichllm snippet "llama 3 8b gguf" --quant Q5_K_M
```

Markdown 输出适用于 GitHub issue、README、Slack、Discord 和博客文章：

```bash
whichllm --markdown
whichllm -m --top 5 --gpu "RTX 4090"
```

JSON 模型字段包含 `fit_type`、`vram_required_bytes`、`vram_available_bytes`、`uses_multi_gpu`、`multi_gpu_effective_vram_bytes`、`estimated_tok_per_sec`、`speed_confidence`、`speed_range_tok_per_sec`、`speed_notes`、`benchmark_source` 和 `benchmark_confidence`。速度范围是规划参考范围，而非实际基准测试结果。

## 集成

### Ollama

使用 JSON 输出将 HuggingFace 模型 ID 映射到你本地的 Ollama 模型名称：

```bash
# 获取排名最高的 HuggingFace 模型 ID
whichllm --top 1 --json | jq -r '.models[0].model_id'

# 获取最佳编码模型 ID
whichllm --profile coding --top 1 --json | jq -r '.models[0].model_id'
```

Ollama 模型名称并不总是与 HuggingFace 仓库 ID 一致，因此通常需要一小步映射后才能使用 `ollama run`。

### Shell 别名

添加到 `.bashrc` / `.zshrc`：

```bash
alias bestllm='whichllm --top 1 --json | jq -r ".models[0].model_id"'
# 用法: ollama run $(bestllm)
```

## 评分机制

每个模型获得 0-100 分的评分。基准测试质量和模型规模构成核心分数；证据置信度和运行时适配进行缩放调整，速度、来源可信度和热度作为微调项。

| 因子 | 效果 | 描述 |
|--------|--------|-------------|
| 基准测试质量 | 核心 | 合并 LiveBench / Artificial Analysis / Aider / Vision / Arena ELO / Open LLM Leaderboard，按来源置信度加权 |
| 模型规模 | 最高 35 分 | `log2` 缩放的世界知识代理（MoE 使用总参数量） |
| 量化等级 | × 惩罚 | 低比特量化以乘法方式折扣 |
| 证据置信度 | ×0.55–1.0 | 无数据/自报 ×0.55，继承 ×0.78，直接匹配满分 |
| 运行时适配 | ×0.50–1.0 | 部分卸载 ×0.72，仅 CPU ×0.50 |
| 速度 | -8 至 +8 | 基于适配类型的 tok/s 下限可用性门控；附带置信度和范围元数据 |
| 来源可信度 | -5 至 +5 | 官方组织加分，已知重新打包者扣分 |
| 热度 | 打破平局 | 下载量/点赞数；证据越强权重越低 |

分数标记：
- **`~`**（黄色）— 无直接基准测试；分数从模型家族继承/插值
- **`!sr`**（亮黄色）— 仅有上传者自报的基准测试分数，未经独立验证
- **`?`**（红色）— 无可用基准测试数据

速度显示：
- **红色** — 生成速度慢（`<4 tok/s`）
- **黄色** — 生成速度勉强（`4-10 tok/s`）
- **绿色** — 生成速度可用（`10-30 tok/s`）
- **亮绿色** — 本地生成速度快（`>=30 tok/s`）
- **`~`**（黄色）— 有估算的 tok/s 范围
- **`?`**（红色）— 低置信度速度估算；后端/运行时敏感度较高

## 文档

- [CLI 参考](docs/cli.md)
- [工作原理](docs/how-it-works.md)
- [评分机制](docs/scoring.md)
- [硬件检测与模拟](docs/hardware.md)
- [运行与代码片段](docs/run-snippet.md)
- [故障排除](docs/troubleshooting.md)

## 工作原理

### 数据管道

1. **模型获取** — 从 HuggingFace API 获取热门模型：
   - 文本生成类（按下载量 + 最近更新时间排序）
   - GGUF 过滤（单独查询以提高覆盖率）
   - 视觉模型（`image-text-to-text`），当使用 `--profile vision` 或 `any` 时
2. **基准测试来源** — *当前层级*（LiveBench、Artificial Analysis Index、Aider）在线合并，以及精选的多模态/视觉索引；*冻结层级*（Open LLM Leaderboard v2、Chatbot Arena ELO）。各层级有独立上限，并结合族谱感知的时效性降权，使过时排行榜不再过度奖励旧一代模型。
3. **基准证据** — 五个解析层级，逐级递减折扣：
   - `direct` — 精确匹配模型 ID
   - `variant` — 去后缀或 -Instruct 变体
   - `base_model` — cardData 中的基础模型
   - `line_interp` — 模型家族内的尺寸感知插值
   - `self_reported` — 上传者声称的评测（大幅折扣）

   当模型的参数量与其家族主要成员差异超过 2 倍时，继承关系会被拒绝，以防止 draft / MTP / abliterated 等分支借用更大基座的分数。
4. **缓存** — 默认位于 `~/.cache/whichllm/`，若设置了 `XDG_CACHE_HOME` 为绝对路径则使用 `$XDG_CACHE_HOME/whichllm/`：
   - `models.json` — 6 小时 TTL
   - `benchmark.json` — 24 小时 TTL

### 排名引擎

1. **硬件检测** — NVIDIA（nvidia-ml-py）、AMD（ROCm/dbgpu）、Intel、Apple Silicon（Metal）、CPU 核心数、内存、磁盘
2. **显存估算** — 权重 + KV 缓存 + 激活值 + 框架开销（约 500MB）
3. **兼容性** — 完全 GPU / 部分卸载 / 仅 CPU；计算能力和操作系统检查
4. **速度** — 基于 GPU 内存带宽、量化等级、后端、适配类型和 MoE 激活参数的 tok/s
5. **评分** — 基准测试（含置信度衰减）、模型规模、量化惩罚、适配类型、速度、热度、来源可信度（官方 vs 重新打包者）
6. **后端过滤** — Apple Silicon 和仅 CPU 限制为 GGUF 以保证稳定性；Linux + NVIDIA 允许 AWQ/GPTQ

### 项目结构

```
src/whichllm/
├── cli.py              # Typer CLI：main, plan, run, snippet, hardware
├── constants.py        # 注册表数据的向后兼容导出
├── data/               # GPU、量化、框架和族谱注册表
├── hardware/
│   ├── detector.py     # 协调 GPU/CPU/RAM 检测
│   ├── nvidia.py       # NVIDIA GPU（通过 nvidia-ml-py）
│   ├── amd.py          # AMD GPU（Linux）
│   ├── apple.py        # Apple Silicon（Metal）
│   ├── cpu.py          # CPU 名称、核心数、AVX 支持
│   ├── memory.py       # 内存和磁盘剩余空间
│   ├── gpu_simulator.py # --gpu 标志：从名称生成模拟 GPU
│   └── types.py        # GPUInfo, HardwareInfo
├── models/
│   ├── fetcher.py      # HuggingFace API、模型解析、evalResults
│   ├── benchmark.py    # Arena ELO、Leaderboard（parquet/rows API）
│   ├── grouper.py      # 按 base_model 和名称进行家族分组
│   ├── cache.py        # 带 TTL 的 JSON 缓存
│   └── types.py        # ModelInfo, GGUFVariant, ModelFamily
├── engine/
│   ├── vram.py         # VRAM = 权重 + KV 缓存 + 激活值 + 开销
│   ├── compatibility.py# 适配类型、磁盘检查、计算能力/操作系统警告
│   ├── performance.py  # 基于带宽的 tok/s
│   ├── quantization.py # 每权重的字节数、质量惩罚、非 GGUF 推理
│   ├── ranker.py       # 评分、证据过滤、配置/匹配
│   └── types.py        # CompatibilityResult
└── output/
    ├── ranking.py      # Rich 硬件和推荐表格
    ├── json_output.py  # 排名、规划和升级的 JSON
    ├── plan.py         # plan 命令显示
    ├── upgrade.py      # upgrade 对比显示
    └── display.py      # 兼容性重导出兼容层
```

## 开发

```bash
git clone https://github.com/Andyyyy64/whichllm.git
cd whichllm
uv sync --dev
uv run whichllm
uv run pytest
```

## 贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

## 支持

如果 whichllm 帮你找到了合适的模型或避免了错误的硬件猜测，欢迎赞助支持。这将帮助项目持续维护：硬件报告、打包、测试、基准测试更新以及更多机器的支持。

无论如何，whichllm 将保持开源。Issue 和 PR 随时欢迎。

觉得有用？一个 GitHub Star 能帮助更多人找到它，我也很想知道它为你的配置推荐了什么。欢迎到 [Issues](https://github.com/Andyyyy64/whichllm/issues) 分享。

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=Andyyyy64/whichllm&type=Date)](https://www.star-history.com/#Andyyyy64/whichllm&Date)

## 系统要求

- Python 3.11+
- NVIDIA GPU 检测通过 `nvidia-ml-py`（默认包含）
- AMD / Apple Silicon 自动检测

## 许可证

MIT
