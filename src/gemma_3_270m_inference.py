# -*- coding: utf-8 -*-
"""
Gemma-3 270M 推論スクリプト

このスクリプトは、best_model_params.ptから訓練済みのGemma-3 270Mモデルパラメータを読み込み、
テキスト生成機能を提供します。
"""
import os
import torch
import tiktoken
import argparse

from gemma_3_270m_slm_scratch import *


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """RoPE（Rotary Position Embedding）のパラメータを計算します。
    
    Args:
        head_dim (int): アテンションヘッドの次元数（偶数である必要があります）
        theta_base (float, optional): 基底周波数。デフォルトは10_000
        context_length (int, optional): コンテキスト長。デフォルトは4096
        dtype (torch.dtype, optional): データ型。デフォルトはtorch.float32
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: cosとsinの事前計算値
            - cos (torch.Tensor): コサイン値、形状 (context_length, head_dim)
            - sin (torch.Tensor): サイン値、形状 (context_length, head_dim)
    
    動作:
        位置エンコーディング用の回転行列パラメータを事前計算し、
        各位置とヘッド次元に対応するcos、sin値を返します。
    
    Raises:
        AssertionError: head_dimが偶数でない場合
    """
    assert head_dim % 2 == 0, "埋め込み次元は偶数である必要があります"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin):
    """RoPE（Rotary Position Embedding）を入力テンソルに適用します。
    
    Args:
        x (torch.Tensor): 入力テンソル、形状 (batch_size, num_heads, seq_len, head_dim)
        cos (torch.Tensor): コサイン値、形状 (context_length, head_dim)
        sin (torch.Tensor): サイン値、形状 (context_length, head_dim)
    
    Returns:
        torch.Tensor: RoPEが適用されたテンソル、形状 (batch_size, num_heads, seq_len, head_dim)
    
    動作:
        入力テンソルを前半と後半に分割し、回転変換を適用して位置情報を
        埋め込みます。これにより、相対的な位置関係を学習できるようになります。
    
    Raises:
        AssertionError: head_dimが偶数でない場合
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "ヘッド次元は偶数である必要があります"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)


GEMMA3_CONFIG_270M = {
    "vocab_size": 50257,
    "context_length": 32_768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
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
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256,
}


def load_model(model_path="src/best_model_params.pt", device="auto"):
    """保存されたパラメータから訓練済みのGemma-3 270Mモデルを読み込みます。
    
    Args:
        model_path (str, optional): モデルパラメータファイルのパス。
                                   デフォルトは"src/best_model_params.pt"
        device (str, optional): 使用するデバイス。"cuda", "cpu", または "auto"。
                                "auto"の場合はCUDAが利用可能ならCUDA、そうでなければCPUを使用
    
    Returns:
        tuple[Gemma3Model, str]: 読み込まれたモデルと使用デバイス名
            - model (Gemma3Model): 評価モードに設定された訓練済みモデル
            - device (str): 実際に使用されるデバイス名（"cuda" または "cpu"）
    
    動作:
        指定されたパスからモデルの状態辞書を読み込み、適切なデバイスに配置し、
        評価モードに設定します。デバイスが"auto"の場合は自動的に最適なデバイスを選択します。
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()
    
    return model, device


def generate_text(model, prompt, max_tokens=200, temperature=1.0, top_k=None, device="cpu"):
    """読み込まれたモデルを使用してテキストを生成します。
    
    Args:
        model (Gemma3Model): テキスト生成に使用する訓練済みモデル
        prompt (str): テキスト生成のための初期プロンプト文字列
        max_tokens (int, optional): 生成する最大トークン数。デフォルトは200
        temperature (float, optional): サンプリング温度（高いほどランダム）。デフォルトは1.0
        top_k (int, optional): Top-Kサンプリングのパラメータ。Noneの場合は無効
        device (str, optional): 計算に使用するデバイス。デフォルトは"cpu"
    
    Returns:
        str: 生成されたテキスト（プロンプトを含む完全な文字列）
    
    動作:
        1. プロンプトをGPT-2トークナイザーでエンコード
        2. モデルを使用して指定された数の新しいトークンを生成
        3. 生成されたトークンをデコードして文字列として返す
        
    温度が高いほどより創造的で多様なテキストが生成され、
    低いほどより一貫性のある予測可能なテキストが生成されます。
    """
    enc = tiktoken.get_encoding("gpt2")
    
    context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(dim=0).to(device)
    
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature=temperature, top_k=top_k)
    
    return enc.decode(generated.squeeze().tolist())


def main():
    """メイン関数：コマンドライン引数を解析してテキスト生成を実行します。
    
    動作:
        1. コマンドライン引数をパース（プロンプト、最大トークン数、温度など）
        2. 指定されたパスからモデルを読み込み
        3. 指定されたパラメータでテキスト生成を実行
        4. 生成されたテキストを標準出力に表示
    
    利用可能な引数:
        --prompt: テキスト生成用のプロンプト文字列
        --max_tokens: 生成する最大トークン数
        --temperature: サンプリング温度
        --top_k: Top-Kサンプリングパラメータ
        --model_path: モデルパラメータファイルのパス
        --device: 使用するデバイス（cuda/cpu/auto）
    """
    parser = argparse.ArgumentParser(description="Gemma-3 270M テキスト生成")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="テキスト生成用のプロンプト")
    parser.add_argument("--max_tokens", type=int, default=200, help="生成する最大トークン数")
    parser.add_argument("--temperature", type=float, default=1.0, help="サンプリング温度")
    parser.add_argument("--top_k", type=int, default=None, help="Top-Kサンプリング")
    parser.add_argument("--model_path", type=str, default="best_model_params.pt", help="モデルパラメータファイルのパス")
    parser.add_argument("--device", type=str, default="auto", help="使用するデバイス (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"エラー: モデルファイル '{args.model_path}' が見つかりません！")
        return
    
    print(f"{args.model_path}からモデルを読み込み中...")
    model, device = load_model(args.model_path, args.device)
    print(f"モデルが{device}上に読み込まれました")
    
    print(f"\nプロンプト '{args.prompt}' に対してテキストを生成中")
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