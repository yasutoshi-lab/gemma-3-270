# Gemma-3-270

## 概要

このリポジトリはgemma-3-270をスクラッチ開発するためのものです

## 構造

```bash
gemma-3-270/
├── README.md
├── best_model_params.pt                                         # モデル
├── log.log                                                      # 手動作成したターミナルlog
├── loss_function.png                                            # プロットしたグラフ
├── pyproject.toml
├── src 
│   └── emma_3_270m_slm_scratch.py                              # 学習用ファイル
│   └── gemma_3_270m_inference.py                               # 推論用ファイル
├── train.bin                                                    # バイナリをトークン化し、トークンを1次元配列として連結したバイナリ
├── training_log.json                                            # 1epochs単位のログ情報
├── uv.lock
└── validation.bin                                               # バイナリをトークン化し、トークンを1次元配列として連結したバイナリ
```

## 学習

```bash
# 通常実行
uv run src/gemma_3_270m_slm_scratch.py
```

## 推論

### 実行例

```bash
# デフォルト設定で推論
uv run src/gemma_3_270m_inference.py

# プロンプトを指定して推論
uv run src/gemma_3_270m_inference.py --prompt "Once upon a time"

# トークン数を指定して推論
uv run src/gemma_3_270m_inference.py --max_tokens 60

# デバイスを指定して推論
uv run src/gemma_3_270m_inference.py --device cuda

# 温度を下げてより確定的な生成
uv run src/gemma_3_270m_inference.py --prompt "A little girl" --temperature 0.8

# Top-kサンプリングを使用
uv run src/gemma_3_270m_inference.py --prompt "Once upon a time" --top_k 50

# 複数パラメータを組み合わせ
uv run src/gemma_3_270m_inference.py --prompt "Grandmother was telling" --max_tokens 100 --temperature 0.9 --top_k 40
```

### 指定可能オプション

| オプション | 型 | デフォルト値 | 説明 |
|-----------|----|-----------|----|
| `--prompt` | str | "Once upon a time" | テキスト生成のためのプロンプト |
| `--max_tokens` | int | 200 | 生成する最大トークン数 |
| `--temperature` | float | 1.0 | サンプリング温度（高いほどランダム） |
| `--top_k` | int | None | Top-kサンプリング（指定数の上位候補から選択） |
| `--model_path` | str | "best_model_params.pt" | モデルパラメータファイルのパス |
| `--device` | str | "auto" | 使用デバイス（cuda/cpu/auto） |


## Push時の注意点
https://qiita.com/kanaya/items/ad52f25da32cb5aa19e6