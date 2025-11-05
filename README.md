# Gemma-3-270

## <u>概要</u>

- google/gemma-3-270mをもとに、アーキテクチャの概念理解の為に作成した学習用リポジトリです
- redditに投稿された学習用ファイルをベースとし、一部実装を変更して利用します[Gemma_3_270m_pre_training](https://www.reddit.com/r/LocalLLaMA/comments/1n0haub/comment/naqnjg9/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)
- 簡易化の為、gemma-3-270mよりパラメーター数が100m程度縮小しています(vocab_size, bias, RMSNorm, RoPE等)
- 初期設定の場合はモデルの学習が充分ではない為、生成精度が極端に低く実用性はありません

<img src=gemma_3_270m.png width=600>

## <u>構成</u>

```bash
gemma-3-270/
├── Dockerfile                  # DockerImage作成用
├── README.md
├── gemma_3_270m.png           # gemma-3-270mアーキテクチャ画像
├── output                     # 成果物のデフォルト出力ディレクトリ
│   ├── config.json            # モデルの構成情報
│   ├── loss_function.png      # 損失の可視化画像
│   ├── model.safetensors      # モデル
│   ├── training_log.json      # 訓練ログ
│   ├── validation_log.json    # 検証ログ
│   └── vocab.json             # 語彙
├── pyproject.toml
├── requirements.txt           # python3-venvのライブラリインストール用
├── src
│   ├── inference.py           # 推論スクリプト
│   ├── model_scratch.py       # 訓練スクリプト
│   ├── test.jsonl             # テストデータセット
│   ├── train.jsonl            # 訓練データセット
│   └── val.jsonl              # 検証データセット
├── use_gpu.png                # デフォルト設定で訓練した場合の消費GPU容量
└── uv.lock
```

## <u>データセット</u>

- HuggingFaceより、[wikipedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)データセットのサブセット"20231101.en"を簡易学習用にサンプリングし、jsonl形式で保存したものです
- ライセンスは[cc-by-sa-3.0](https://spdx.org/licenses/CC-BY-SA-3.0)を継承します

※ データセットのサンプリング
```python
train_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:10000]")
val_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[10000:11000]")
test_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[81000:82000]")
```

## <u>検証環境</u>

### ハードウェア要件

- GPU: rtx4090 16GB
- RAM: 62GB
- DISK: 2TB
- CPU: i9-14900HX

### システム要件

- OS: Ubuntu22.04.5
- Architecture: x86_64
- UV: 0.8.4
- CUDA 12.9

## <u>セットアップ</u>

- 仮想環境の作成と適用

```bash
# リポジトリのクローン
git clone https://github.com/yasutoshi-lab/gpt-oss.git

# ディレクトリ変更
cd gpt-oss/

# 仮想環境作成
uv venv --python 3.12.3

# 仮想環境適用
source .venv/bin/activate

# ライブラリの同期 
uv sync
```

- 訓練の実行

```json
// デフォルトモデルパラメーター
{
    "vocab_size": 50257,              // 語彙サイズ
    "context_length": 32_768,         // コンテキスト長
    "emb_dim": 640,                   // 埋め込み次元
    "n_heads": 4,                     // アテンションヘッド数
    "n_layers": 18,                   // レイヤー数
    "hidden_dim": 2048,               // 隠れ層次元
    "head_dim": 256,                  // ヘッド次元
    "qk_norm": True,                  // QK正規化
    "n_kv_groups": 1,                 // KVグループ数
    "rope_local_base": 10_000.0,      // RoPEローカルベース
    "rope_base": 1_000_000.0,         // RoPEベース
    "sliding_window": 512,            // スライディングウィンドウ
      "layer_types": [                // レイヤータイプ
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ],
    "dtype": torch.bfloat16,          // データ型
    "query_pre_attn_scalar": 256,     // クエリスケーラー
}
```

```json
// デフォルトコマンドライン引数
{
  "--output_dir": "output",               // 出力先ディレクトリ
  "--learning_rate": 0.0001,              // 学習率
  "--max_steps": 10000,                   // 最大ステップ数(global_stepより設定値が小さい場合のみ有効)
  "--warmup_steps": 100,                  // ウォームアップステップ数
  "--min_lr": 0.0005,                     // 最小学習率
  "--eval_iters": 200,                    // 検証ステップ間隔
  "--gradient_accumulation_steps": 4,     // 勾配累積
  "--train_batch_size": 1,                // 訓練バッチサイズ
  "--val_batch_size": 1,                  // 検証バッチサイズ
  "--max_length": 2048,                   // コーパスの最大長
  "--stride": 2048,                       // コーパスのオーバーラップ間隔
  "--add_eos_between_documents": true,    // コーパス毎のeos付与
  "--eos_token": "<|endoftext|>",         // eos_token(デフォルトのエンコーダーがgpt2の為)
  "--logging_steps": 1,                   // 訓練ログの出力間隔
  "--weight_decay": 0.1,                  // 最適化関数の重み減衰率
  "--betas": [0.9, 0.95],                 // 最適化関数のβスケジューラー値
  "--seed": 42,                           // 乱数シード値
  "--tiktoken_encoder_name": "gpt2",      // エンコーダー
  "--train_dataset_path": "src/train.jsonl",  // 訓練データセットのパス
  "--val_dataset_path": "src/val.jsonl".  // 検証データセットのパス
}
```

```bash
# 訓練スクリプトの実行
uv run src/model_scratch.py
```

- 推論の実行

```bash
# 推論スクリプトの実行
uv run src/inference.py
```

## <u>再現モデル構造</u>

- パラメーター数：164,654,976(164.655M)
- データ型: bfloa16

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

## <u>ベースモデル構造</u>

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

## <u>環境構築: UV</u>

```bash
# インストール
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# インストール確認
uv --version
```

## <u>環境構築: NVIDIA-Driver, CUDA-Toolkit</u>

※下記スクリプトは検証環境のシステム要件に従ったものです。実際の環境に合わせてインストールしてください  
[CUDA Toolkit 12.9 Downloads](https://developer.nvidia.com/cuda-12-9-0-download-archive)

```bash
# CUDA Toolkit Installer
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

# NVIDIA Driver Installer
sudo apt-get install -y nvidia-open
```

## <u>Dockerセットアップ</u>

- コンテナーイメージの作成と実行

```bash
# コンテナーイメージの作成
docker build -t gemma-3-270 .

# コンテナーの実行
docker run -it --gpus all gemma-3-270 /bin/bash
```

- 訓練の実行

```bash
#  訓練スクリプトの実行
python3 src/model_scratch.py
```

- 推論の実行

```bash
# 推論スクリプトの実行
python3 src/inference.py
```

## <u>環境構築: Docker</u>

※下記スクリプトは検証環境のシステム要件に従ったものです。実際の環境に合わせてインストールしてください  
[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

```bash
# install docker packages
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# change permission
sudo groupadd docker
sudo usermod -aG docker $USER # Please sign out and sign in.
```

## <u>環境構築: NVIDIA-Container-Toolkit</u>

※下記スクリプトは検証環境のシステム要件に従ったものです。実際の環境に合わせてインストールしてください   
[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
# Install the prerequisites for the instructions below:
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   curl \
   gnupg2

# Configure the production repository:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Optionally, configure the repository to use experimental packages:
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update the packages list from the repository:
sudo apt-get update

# Install the NVIDIA Container Toolkit packages:
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.0-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

```bash
# create service config
sudo touch /etc/docker/daemon.json

# edit config
sudo vim /etc/docker/daemon.json
```

```json
// /etc/docker/daemon.json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```

```bash
# apply service config
sudo systemctl daemon-reload

# restart service
sudo systemctl restart docker
```
