# -*- coding: utf-8 -*-
"""
Gemma-3 270M Inference Script

This script loads the trained Gemma-3 270M model parameters from best_model_params.pt
and provides text generation capabilities.
"""
import os
import torch
import tiktoken
import argparse

from gemma_3_270m_slm_scratch import *


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

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
    """Load the trained Gemma-3 270M model from saved parameters."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()
    
    return model, device


def generate_text(model, prompt, max_tokens=200, temperature=1.0, top_k=None, device="cpu"):
    """Generate text using the loaded model."""
    enc = tiktoken.get_encoding("gpt2")
    
    context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(dim=0).to(device)
    
    with torch.no_grad():
        generated = model.generate(context, max_tokens, temperature=temperature, top_k=top_k)
    
    return enc.decode(generated.squeeze().tolist())


def main():
    parser = argparse.ArgumentParser(description="Gemma-3 270M Text Generation")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--model_path", type=str, default="best_model_params.pt", help="Path to model parameters")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        return
    
    print(f"Loading model from {args.model_path}...")
    model, device = load_model(args.model_path, args.device)
    print(f"Model loaded on {device}")
    
    print(f"\nGenerating text for prompt: '{args.prompt}'")
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