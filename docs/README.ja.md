# whichllm

**あなたのPCで動く最強のローカルLLMを見つけるCLIツール。**

GPU/CPU/RAMを自動検出し、HuggingFaceの人気モデルからハードウェアに合った最適なモデルをランキング表示します。

[English version](../README.md)

![demo](../assets/demo.png)

## インストール

### pipx (推奨)

```bash
pipx install whichllm
```

### pip

```bash
pip install whichllm
```

### 開発用

```bash
git clone https://github.com/Andyyyy64/whichllm.git
cd whichllm
uv sync --dev
uv run whichllm
```

## 使い方

```bash
# 自動検出して最適なモデルを表示
whichllm

# GPU をシミュレート (購入検討時など)
whichllm --gpu "RTX 4090"
whichllm --gpu "RTX 5090"

# CPU のみモードで実行
whichllm --cpu-only

# 結果を増やす / フィルタ
whichllm --top 20
whichllm --quant Q4_K_M
whichllm --min-speed 30
whichllm --evidence base   # id一致 + base_model一致まで許可
whichllm --evidence strict # id完全一致のみ（--direct と同じ）
whichllm --direct

# JSON 出力
whichllm --json

# キャッシュを無視して再取得
whichllm --refresh

# ハードウェア情報だけ表示
whichllm hardware
```

## スコアの見方

各モデルに 0〜100 のスコアが付きます。

| 要素 | 配点 | 説明 |
|------|------|------|
| モデルサイズ | 0-40 | パラメータ数が大きいほど高品質 |
| ベンチマーク | 0-10 | Arena ELO / Open LLM Leaderboard のスコア |
| 推論速度 | 0-20 | tok/s が高いほど実用的 |
| ソース信頼度 | -5〜+5 | 公式 org はボーナス、リパッケージはペナルティ |
| 人気度 | 0-3 | ダウンロード数・いいね数 |

スコア横のマーカー:
- **`~`** (黄色) — ベンチマークがまだ公開されていない新しいモデル。同シリーズの推定値を使用
- **`?`** (黄色) — ベンチマークデータなし

## 仕組み

### データ取得

1. **HuggingFace API** から人気テキスト生成モデル + GGUF + マルチモーダルを取得 (計 ~900 モデル)
2. **Chatbot Arena ELO** と **Open LLM Leaderboard** からベンチマークスコアを取得・正規化
3. 24時間キャッシュ (`~/.cache/whichllm/`)

### ランキングエンジン

1. **ハードウェア検出** — GPU (NVIDIA/AMD/Apple), CPU, RAM, ディスクを自動検出
2. **VRAM 見積もり** — モデルサイズ + 量子化 + KV キャッシュから必要 VRAM を計算
3. **互換性チェック** — Full GPU / Partial Offload / CPU-only の判定
4. **推論速度推定** — GPU メモリ帯域幅ベースの tok/s 推定
5. **スコア計算** — サイズ、ベンチマーク、速度、信頼度を総合評価
6. **ファミリー重複排除** — 同じモデルの GGUF バリアントやバージョン違いを統合

### プロジェクト構成

```
src/whichllm/
├── cli.py              # typer CLI エントリポイント
├── constants.py        # GPU 帯域幅テーブル、量子化定数
├── hardware/           # ハードウェア検出 (NVIDIA, AMD, Apple, CPU, RAM)
│   └── gpu_simulator.py  # --gpu フラグ用 GPU シミュレータ
├── models/
│   ├── fetcher.py      # HuggingFace API からモデル取得
│   ├── benchmark.py    # ベンチマークスコア取得 (Arena + Leaderboard)
│   ├── grouper.py      # モデルファミリー分類・重複排除
│   └── cache.py        # JSON キャッシュ
├── engine/
│   ├── vram.py         # VRAM 必要量の推定
│   ├── compatibility.py # ハードウェア互換性チェック
│   ├── performance.py  # 推論速度推定
│   └── ranker.py       # スコア計算・ランキング
└── output/
    └── display.py      # Rich テーブル表示
```

## 動作環境

- Python 3.11+
- NVIDIA GPU 検出は `nvidia-ml-py` で自動対応（デフォルトで同梱）
- AMD / Apple Silicon は自動検出

## ライセンス

MIT
