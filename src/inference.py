import torch
import json
import tiktoken
import argparse
from safetensors.torch import load_file
from model_scratch import Gemma3Model, GEMMA3_CONFIG_270M

def generate_text(args):
    """テキスト生成を実行
    
    Args:
        args (argparse.Namespace): コマンドライン引数
    
    Returns:
        str: 生成されたテキスト
        
    Behavior:
        1. gpt-2エンコーダーの読み込み
        2. Gemma3モデルアーキテクチャの読み込み
        3. モデル重みの読み込み
        4. 入力文のエンコード
        5. モデルを使用してテキスト生成
        6. 生成されたトークンをデコードしてテキストを返す
    """
    enc = tiktoken.get_encoding("gpt2")
    model = Gemma3Model(GEMMA3_CONFIG_270M)
    model.load_state_dict(load_file(args.model_path, device=args.device))
    context = (torch.tensor(enc.encode_ordinary(args.prompt)).unsqueeze(dim = 0))
    generate_tokens = model.generate(context, args.max_tokens)
    return enc.decode(generate_tokens.squeeze().tolist())

def main():
    """コマンドライン引数を取得
    
    Behavior: 
        uv run src/inference.py \
        --prompt "Language Model" \
        --max_tokens 200 \
        --model_path "output/model.safetensors" \
        --device "cpu"
    """
    parser = argparse.ArgumentParser(description="Pre-Trainedモデルのテキスト生成")
    parser.add_argument("--prompt", type=str, default="Neural Networks", help="テキスト生成用のプロンプト")
    parser.add_argument("--max_tokens", type=int, default=200, help="生成する最大トークン数")
    parser.add_argument("--model_path", type=str, default="output/model.safetensors", help="モデルパラメータファイルのパス")
    parser.add_argument("--device", type=str, default="cpu", help="使用するデバイス (cuda/cpu)")
    args = parser.parse_args()
    print(generate_text(args))

if __name__ == "__main__":
    main()