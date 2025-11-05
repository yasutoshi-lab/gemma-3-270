# Gemma-3-270

## 概要

- google/gemma-3-270mをもとに、アーキテクチャの概念理解の為に作成した学習用リポジトリです
- redditに投稿された学習用ファイルを参考に、一部実装を変更して利用します[Gemma_3_270m_pre_training](https://www.reddit.com/r/LocalLLaMA/comments/1n0haub/comment/naqnjg9/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)



## ベースモデルアーキテクチャ

<img src=gemma_3_270m.png width=600>

```txt
Gemma3ForCausalLM(
  (model): Gemma3TextModel(
    (embed_tokens): Gemma3TextScaledWordEmbedding(262144, 640, padding_idx=0)
    (layers): ModuleList(
      (0-17): 18 x Gemma3DecoderLayer(
        (self_attn): Gemma3Attention(
          (q_proj): Linear(in_features=640, out_features=1024, bias=False)
          (k_proj): Linear(in_features=640, out_features=256, bias=False)
          (v_proj): Linear(in_features=640, out_features=256, bias=False)
          (o_proj): Linear(in_features=1024, out_features=640, bias=False)
          (q_norm): Gemma3RMSNorm((256,), eps=1e-06)
          (k_norm): Gemma3RMSNorm((256,), eps=1e-06)
        )
        (mlp): Gemma3MLP(
          (gate_proj): Linear(in_features=640, out_features=2048, bias=False)
          (up_proj): Linear(in_features=640, out_features=2048, bias=False)
          (down_proj): Linear(in_features=2048, out_features=640, bias=False)
          (act_fn): PytorchGELUTanh()
        )
        (input_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
        (post_attention_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
        (pre_feedforward_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
        (post_feedforward_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
      )
    )
    (norm): Gemma3RMSNorm((640,), eps=1e-06)
    (rotary_emb): Gemma3RotaryEmbedding()
    (rotary_emb_local): Gemma3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=640, out_features=262144, bias=False)
)
```


## 再現モデルアーキテクチャ

```txt
Gemma3Model(
  (tok_emb): Embedding(50257, 640)
  (blocks): ModuleList(
    (0-17): 18 x TransformerBlock(
      (att): GroupedQueryAttention(
        (W_query): Linear(in_features=640, out_features=1024, bias=False)
        (W_key): Linear(in_features=640, out_features=256, bias=False)
        (W_value): Linear(in_features=640, out_features=256, bias=False)
        (out_proj): Linear(in_features=1024, out_features=640, bias=False)
        (q_norm): RMSNorm()
        (k_norm): RMSNorm()
      )
      (ff): FeedForward(
        (fc1): Linear(in_features=640, out_features=2048, bias=False)
        (fc2): Linear(in_features=640, out_features=2048, bias=False)
        (fc3): Linear(in_features=2048, out_features=640, bias=False)
      )
      (input_layernorm): RMSNorm()
      (post_attention_layernorm): RMSNorm()
      (pre_feedforward_layernorm): RMSNorm()
      (post_feedforward_layernorm): RMSNorm()
    )
  )
  (final_norm): RMSNorm()
  (out_head): Linear(in_features=640, out_features=50257, bias=False)
)
```














## 構造

```bash
gemma-3-270/
├── README.md
├── pyproject.toml
├── src 
│   └── model_scratch.py                              # 学習用ファイル
│   └── inference.py                               # 推論用ファイル                                         # 1epochs単位のログ情報
├── uv.lock                                         # バイナリをトークン化し、トークンを1次元配列として連結したバイナリ
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