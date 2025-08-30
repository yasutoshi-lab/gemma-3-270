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
│   ├── gemma_3_270_m_small_language_model_scratch_final.ipynb
│   └── gemma_3_270_m_small_language_model_scratch_final.py 
├── train.bin                                                    # バイナリをトークン化し、トークンを1次元配列として連結したバイナリ
├── training_log.json                                            # 1epochs単位のログ情報
├── uv.lock
└── validation.bin                                               # バイナリをトークン化し、トークンを1次元配列として連結したバイナリ
```

## Push時の注意点
https://qiita.com/kanaya/items/ad52f25da32cb5aa19e6