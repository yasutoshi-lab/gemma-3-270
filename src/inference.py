import os
import torch
import tiktoken
import argparse

from model_scratch import *


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """RoPE（Rotary Position Embedding）のパラメータを計算
    
    Args:
        head_dim (int): アテンションヘッドの次元数
        theta_base (float, optional): 基底周波数
        context_length (int, optional): コンテキスト長
        dtype (torch.dtype, optional): データ型
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: cosとsinの事前計算値
            - cos (torch.Tensor): コサイン値、形状 (context_length, head_dim)
            - sin (torch.Tensor): サイン値、形状 (context_length, head_dim)
    
    Behavior:
        位置エンコーディング用の回転行列パラメータを事前計算し、
        各位置とヘッド次元に対応するcos、sin値を返す
    
    Raises:
        AssertionError: head_dimが偶数でない
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin):
    """RoPE（Rotary Position Embedding）を入力テンソルに適用
    
    Args:
        x (torch.Tensor): 入力テンソル
        cos (torch.Tensor): コサイン値
        sin (torch.Tensor): サイン値
    
    Returns:
        torch.Tensor: RoPEが適用されたテンソル、形状 (batch_size, num_heads, seq_len, head_dim)
    
    Behavior:
        入力テンソルを前半と後半に分割し、回転変換を適用して位置情報を埋め込む
    
    Raises:
        AssertionError: head_dimが偶数でない
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)


def load_config(config_path="output/config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_model(model_path="output/best_model_params.pt", device="auto"):
    """保存されたパラメータから訓練済みのGemma-3 270Mモデルを読み込み
    
    Args:
        model_path (str, optional): モデルパラメータファイルのパス。
                                   デフォルトは"output/best_model_params.pt"
        device (str, optional): 使用するデバイス。"cuda", "cpu", または "auto"
                                "auto"の場合はCUDAが利用可能ならCUDA、そうでなければCPUを使用
    
    Returns:
        tuple[Gemma3Model, str]: 読み込まれたモデルと使用デバイス名
            - model (Gemma3Model): 評価モードに設定された訓練済みモデル
            - device (str): 実際に使用されるデバイス名
    
    Behavior:
        指定されたパスからモデルの状態辞書を読み込み、適切なデバイスに配置し、
        評価モードに設定。デバイスが"auto"の場合は自動的に最適なデバイスを選択。
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = load_config(args.config_path)
    model = Gemma3Model(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()
    
    return model, device


def generate_text(model, prompt, max_tokens=200, temperature=1.0, top_k=None, device="cpu"):
    """読み込まれたモデルを使用してテキストを生成
    
    Args:
        model (Gemma3Model): テキスト生成に使用する訓練済みモデル
        prompt (str): テキスト生成のための初期プロンプト文字列
        max_tokens (int, optional): 生成する最大トークン数
        temperature (float, optional): サンプリング温度（高いほどランダム）
        top_k (int, optional): Top-Kサンプリングのパラメータ
        device (str, optional): 計算に使用するデバイス
    
    Returns:
        str: 生成されたテキスト
    
    Behavior:
        1. プロンプトをトークナイザーでエンコード
        2. モデルを使用して指定された数の新しいトークンを生成
        3. 生成されたトークンをデコードして文字列として返す
    """
    enc = tiktoken.get_encoding("gpt2")
    
    context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(dim=0).to(device)
    
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature=temperature, top_k=top_k)
    
    return enc.decode(generated.squeeze().tolist())


def main():
    """メイン関数：コマンドライン引数を解析してテキスト生成を実行
    
    Behavior:
        1. コマンドライン引数をパース
        2. 指定されたパスからモデルを読み込み
        3. 指定されたパラメータでテキスト生成を実行
        4. 生成されたテキストを標準出力に表示
    
    Available arguments:
        --prompt: テキスト生成用のプロンプト文字列
        --max_tokens: 生成する最大トークン数
        --temperature: サンプリング温度
        --top_k: Top-Kサンプリングパラメータ
        --model_path: モデルパラメータファイルのパス
        --device: 使用するデバイス（cuda/cpu/auto）
    """
    parser = argparse.ArgumentParser(description="Gemma-3 270M テキスト生成")
    parser.add_argument("--prompt", type=str, default="Neural Networks", help="テキスト生成用のプロンプト")
    parser.add_argument("--max_tokens", type=int, default=200, help="生成する最大トークン数")
    parser.add_argument("--temperature", type=float, default=1.0, help="サンプリング温度")
    parser.add_argument("--top_k", type=int, default=None, help="Top-Kサンプリング")
    parser.add_argument("--model_path", type=str, default="output/best_model_params.pt", help="モデルパラメータファイルのパス")
    parser.add_argument("--config_path", type=str, default="output/config.json", help="モデル設定ファイルのパス")
    parser.add_argument("--device", type=str, default="auto", help="使用するデバイス (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: '{args.model_path}' not found")
        return
    
    print(f"Loading model from {args.model_path}...")
    model, device = load_model(args.model_path, args.device)
    print(f"Model loaded on {device}")
    
    print(f"\nGenerating text for prompt '{args.prompt}'")
    print("=" * 50)
    
    generated_text = generate_text(
        model, 
        args.prompt, 
        args.max_tokens, 
        args.temperature, 
        args.top_k, 
        device
    )
    
    print(generated_text)


if __name__ == "__main__":
    main()