
import os
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset



# train_ds = load_dataset("json", data_files="train_dataset_25000.jsonl")
# validation_ds = load_dataset("json", data_files="validation_dataset_2500.jsonl")



# def process(example):
#     """テキストサンプルをトークンIDに変換
    
#     Args:
#         example (dict): 'text'キーを含む辞書
#             - text (str): トークン化するテキスト文字列
    
#     Returns:
#         dict: トークン化された結果を含む辞書
#             - ids (list[int]): トークンIDのリスト
#             - len (int): トークンIDの数
    
#     Behavior:
#         入力テキストをGPT-2エンコーダーを使用してトークンIDに変換
#         特殊トークンを無視してIDとその長さを返す
#     """
#     ids = enc.encode_ordinary(example['text'])  # 無視してエンコード
#     out = {'ids': ids, 'len': len(ids)}
#     return out

# if not os.path.exists("output/train.bin"):
#     tokenized = train_ds.map(
#         process,
#         remove_columns=['text'],
#         desc="tokenizing the splits",
#         num_proc=4,
#         )
#     # 各データセットのIDを結合して大きなファイルを作成
#     arr_len = np.sum(tokenized['len'], dtype=np.uint64)
#     filename = f'output/train.bin'
#     dtype = np.uint16 # (enc.max_token_value == 50256 < 2**16)
#     arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
#     arr[:] = np.concatenate(tokenized['ids'])
#     arr.flush()

# if not os.path.exists("output/validation.bin"):
#     tokenized = validation_ds.map(
#         process,
#         remove_columns=['text'],
#         desc="tokenizing the splits",
#         num_proc=4,
#         )
#     arr_len = np.sum(tokenized['len'], dtype=np.uint64)
#     filename = f'output/validation.bin'
#     dtype = np.uint16 # (enc.max_token_value == 50256 < 2**16)
#     arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
#     arr[:] = np.concatenate(tokenized['ids'])
#     arr.flush()

# # block_size = コンテキストウィンドウ
# def get_batch(split):
#     """指定された分割（訓練または検証）からバッチデータを取得
    
#     Args:
#         split (str): データ分割の種類
    
#     Returns:
#         tuple[torch.Tensor, torch.Tensor]: 入力と目標出力のテンソルペア
#             - x (torch.Tensor): 入力シーケンス、形状 (batch_size, block_size)
#             - y (torch.Tensor): 目標出力シーケンス、形状 (batch_size, block_size)
    
#     Behavior:
#         メモリリークを回避するため、各バッチでnp.memmapを再作成
#         指定された分割からランダムなシーケンスを抽出してバッチを作成
#         GPUが利用可能な場合は、非同期転送のためにメモリをピン留め
#     """
#     # メモリリークを回避するため、各バッチでnp.memmapを再作成
#     # 参考: https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
#     if split == 'train':
#         data = np.memmap('output/train.bin', dtype=np.uint16, mode='r')
#     else:
#         data = np.memmap('output/validation.bin', dtype=np.uint16, mode='r')
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
#     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
#     if device_type == 'cuda':
#         # 配列x,yをピン留めし、GPUへ非同期移動（non_blocking=True)
#         x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
#     else:
#         x, y = x.to(device), y.to(device)
#     return x, y

from dataloader import create_dataloader_v1

import torch

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
        AssertionError: head_dim is not even
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 逆周波数を計算
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # 位置インデックスを生成
    positions = torch.arange(context_length, dtype=dtype)

    # 角度を計算
    angles = positions[:, None] * inv_freq[None, :]  # 形状: (context_length, head_dim // 2)

    # head_dimに合わせて角度を拡張
    angles = torch.cat([angles, angles], dim=1)  # 形状: (context_length, head_dim)

    # サインとコサインを事前計算
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin):
    """RoPE（Rotary Position Embedding）を入力テンソルに適用
    
    Args:
        x (torch.Tensor): 入力テンソル、形状 (batch_size, num_heads, seq_len, head_dim)
        cos (torch.Tensor): コサイン値、形状 (context_length, head_dim)
        sin (torch.Tensor): サイン値、形状 (context_length, head_dim)
    
    Returns:
        torch.Tensor: RoPEが適用されたテンソル、形状 (batch_size, num_heads, seq_len, head_dim)
    
    Behavior:
        入力テンソルを前半と後半に分割し、回転変換を適用して位置情報を埋め込む
    
    Raises:
        AssertionError: head_dim is not even
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # xを前半と後半に分割
    x1 = x[..., : head_dim // 2]  # 前半
    x2 = x[..., head_dim // 2 :]  # 後半

    # sinとcosの形状を調整
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # 回転変換を適用
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # cosとsin回転を適用した後は低精度を使用しても問題ない
    return x_rotated.to(dtype=x.dtype)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import numpy as np
from tqdm.auto import tqdm
from contextlib import nullcontext
import os
import json

class RMSNorm(nn.Module):
    """Root Mean Square正規化レイヤー（Gemma3実装）。
    
    標準的なLayerNormの代わりにRMS正規化を使用し、より計算効率的な正規化を提供
    Gemma3では、ゼロ中心の重みを保存し、順伝播時に(1 + weight)を使用
    """

    def __init__(self, emb_dim, eps=1e-6, bias=False):
        """RMSNormレイヤーを初期化
        
        Args:
            emb_dim (int): 埋め込み次元数
            eps (float, optional): 数値安定性のための小さな値
            bias (bool, optional): バイアス項を使用するかどうか。デフォルトはFalse
        
        Behavior:
            スケールパラメータをゼロで初期化し、オプションでシフトパラメータを作成
        """
        super().__init__()
        self.eps = eps
        # Gemma3はゼロ中心の重みを保存し、順伝播時に(1 + weight)を使用
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        """RMS正規化を適用します。
        
        Args:
            x (torch.Tensor): 入力テンソル、形状 (..., emb_dim)
        
        Returns:
            torch.Tensor: 正規化された出力テンソル、入力と同じ形状とデータ型 (..., emb_dim)
        
        Behavior:
            HuggingFace Gemma3に合わせて、float32で正規化を計算し、
            その後(1 + w)でスケーリング。最後に元のデータ型に戻す
        """
        # HuggingFace Gemma3に合わせる: float32で正規化を計算し、その後(1 + w)でスケール
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())

        if self.shift is not None:
            out = out + self.shift.float()

        return out.to(input_dtype)



class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention（GQA）
    
    標準的なMulti-Head Attentionの効率化版で、キーと値のヘッド数を削減することで
    計算量とメモリ使用量を削減。複数のクエリヘッドが同じキー・値ヘッドを共有
    """
    
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False,
        query_pre_attn_scalar=None, dtype=None,
    ):
        """GroupedQueryAttentionレイヤーを初期化
        
        Args:
            d_in (int): 入力特徴量の次元数
            num_heads (int): アテンションヘッド数
            num_kv_groups (int): キー・値のグループ数
            head_dim (int, optional): 各ヘッドの次元数。Noneの場合はd_in/num_heads
            qk_norm (bool, optional): クエリとキーに正規化を適用するかどうか
            query_pre_attn_scalar (float, optional): クエリのスケーリング係数
            dtype (torch.dtype, optional): 線形層のデータ型
        
        Behavior:
            GQAのための線形変換層を初期化し、オプションで正規化とスケーリングを設定
            
        Raises:
            AssertionError: num_heads is not divisible by num_kv_groups
                           or head_dim is None and d_in is not divisible by num_heads
        """
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "head_dim is not set and d_in is not divisible by num_heads"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        if query_pre_attn_scalar is not None:
            self.scaling = (query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = (head_dim) ** -0.5


    def forward(self, x, mask, cos, sin):
        """Grouped Query Attentionの順伝播を実行
        
        Args:
            x (torch.Tensor): 入力テンソル、形状 (batch_size, num_tokens, d_in)
            mask (torch.Tensor): アテンションマスク、形状 (num_tokens, num_tokens)
            cos (torch.Tensor): RoPE用のコサイン値
            sin (torch.Tensor): RoPE用のサイン値
        
        Returns:
            torch.Tensor: アテンション後の出力、形状 (batch_size, num_tokens, d_out)
        
        Behavior:
            1. クエリ、キー、値への線形変換を適用
            2. オプションで正規化を実行
            3. RoPE（Rotary Position Embedding）を適用
            4. GQAのためにキーと値を拡張
            5. スケールドドット積アテンションを計算
            6. 出力射影を適用
        """
        b, num_tokens, _ = x.shape

        # 線形変換を適用
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # 形状変更
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # オプションの正規化
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # RoPEを適用
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # ヘッド数に合わせてKとVを拡張
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # クエリをスケール
        queries = queries * self.scaling

        # アテンション計算
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)

class FeedForward(nn.Module):
    """ゲート付きフィードフォワードネットワーク（Gemma3スタイル）
    
    SwiGLU活性化関数を使用した2つの並列な線形変換を持つフィードフォワード層
    一方の経路はGELU活性化を適用し、もう一方はゲートとして機能
    """
    
    def __init__(self, cfg):
        """FeedForwardレイヤーを初期化
        
        Args:
            cfg (dict): 設定辞書
                - emb_dim (int): 埋め込み次元数
                - hidden_dim (int): 隠れ層の次元数
                - dtype (torch.dtype): データ型
        
        Behavior:
            3つの線形層を初期化：
            - fc1: 入力から隠れ層への変換（GELU用）
            - fc2: 入力から隠れ層への変換（ゲート用）
            - fc3: 隠れ層から出力への変換
        """
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        """ゲート付きフィードフォワードの順伝播を実行
        
        Args:
            x (torch.Tensor): 入力テンソル、形状 (batch_size, seq_len, emb_dim)
        
        Returns:
            torch.Tensor: 出力テンソル、形状 (batch_size, seq_len, emb_dim)
        
        Behavior:
            SwiGLU活性化を実装：x_output = fc3(GELU(fc1(x)) * fc2(x))
            一方の経路にGELU活性化を適用し、もう一方をゲートとして要素積を計算
        """
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    """シングルTransformerブロック（Gemma3スタイル）
    
    アテンション機構とフィードフォワードネットワークを含む完全なTransformerブロック
    スライディングウィンドウアテンションまたは全アテンションをサポート
    """

    def __init__(self, cfg: dict, attn_type: str):
        """シングルTransformerブロックを初期化
        
        Args:
            cfg (dict): モデル設定辞書
                - emb_dim (int): 埋め込み次元数
                - n_heads (int): アテンションヘッド数
                - n_kv_groups (int): キー・値のグループ数
                - head_dim (int): ヘッド次元数
                - qk_norm (bool): クエリ/キー正規化の有無
                - query_pre_attn_scalar (float): クエリスケーリング係数
                - dtype (torch.dtype): データ型
            attn_type (str): アテンションの種類（'sliding_attention' または 'full_attention'）
        
        Behavior:
            指定された設定でアテンション、フィードフォワード、正規化層を初期化
            Gemma3スタイルの4つのRMSNorm層（入力、アテンション後、FF前、FF後）を使用
        """
        super().__init__()
        self.attn_type = attn_type

        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"],
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(
        self,
        x,
        mask_global,
        mask_local,
        cos_global,
        sin_global,
        cos_local,
        sin_local,
    ):
        """Transformerブロックの順伝播を実行
        
        Args:
            x (torch.Tensor): 入力テンソル、形状 (batch_size, seq_len, emb_dim)
            mask_global (torch.Tensor): 全アテンション用マスク
            mask_local (torch.Tensor): スライディングアテンション用マスク
            cos_global (torch.Tensor): 全アテンション用RoPEコサイン値
            sin_global (torch.Tensor): 全アテンション用RoPEサイン値
            cos_local (torch.Tensor): スライディングアテンション用RoPEコサイン値
            sin_local (torch.Tensor): スライディングアテンション用RoPEサイン値
        
        Returns:
            torch.Tensor: 出力テンソル、形状 (batch_size, seq_len, emb_dim)
        
        Behavior:
            1. アテンションブロック: LayerNorm → Attention → LayerNorm → 残差接続
            2. フィードフォワードブロック: LayerNorm → FeedForward → LayerNorm → 残差接続
            アテンションタイプに応じて適切なマスクとRoPEパラメータを使用
        """
        # アテンションブロックの残差接続
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            attn_mask = mask_local
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = mask_global
            cos = cos_global
            sin = sin_global

        x_attn = self.att(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        # フィードフォワードブロックの残差接続
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x

class Gemma3Model(nn.Module):
    """完全なGemma3モデル。
    
    トークン埋め込み、複数のTransformerブロック、最終正規化、出力ヘッドを含む
    完全な言語モデル。スライディングウィンドウアテンションと全アテンションの
    組み合わせをサポート
    """
    
    def __init__(self, cfg):
        """完全なGemma3モデルを初期化
        
        Args:
            cfg (dict): モデル設定辞書
                - vocab_size (int): 語彙サイズ
                - emb_dim (int): 埋め込み次元数
                - n_layers (int): Transformerブロックの数
                - layer_types (list[str]): 各層のアテンションタイプのリスト
                - head_dim (int): アテンションヘッドの次元数
                - rope_local_base (float): スライディングアテンション用RoPE基底周波数
                - rope_base (float): 全アテンション用RoPE基底周波数
                - context_length (int): コンテキスト長
                - dtype (torch.dtype): データ型
                - sliding_window (int): スライディングウィンドウのサイズ
        
        Behavior:
            全モデルコンポーネントを初期化し、各アテンションタイプ用の
            RoPEパラメータを事前計算してバッファに登録
            
        Raises:
            AssertionError: layer_types is None or length is not equal to n_layers
        """
        super().__init__()
        assert cfg["layer_types"] is not None and len(cfg["layer_types"]) == cfg["n_layers"]

        # メインモデルパラメータ
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, attn_type)for attn_type in cfg["layer_types"]
        ])

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        self.cfg = cfg

        # 再使用可能なユーティリティ
        cos_local, sin_local = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, seq_len, device):
        """アテンション用のマスクを作成
        
        Args:
            seq_len (int): シーケンスの長さ
            device (torch.device): デバイス
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: グローバルマスクとローカルマスク
                - mask_global (torch.Tensor): 全アテンション用マスク（因果関係のみ）
                - mask_local (torch.Tensor): スライディングアテンション用マスク
        
        Behavior:
            1. mask_global: 因果関係を維持するため、将来のトークンをマスク
            2. mask_local: スライディングウィンドウを超えた過去のトークンもマスク
            Trueの箱所はマスクされ（-infに設定）、Falseの箱所は有効
        """
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

        # mask_global (将来はマスク: j > i)
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 1 1 1 1 1 1 1
        #     1:  0 0 1 1 1 1 1 1
        #     2:  0 0 0 1 1 1 1 1
        #     3:  0 0 0 0 1 1 1 1
        #     4:  0 0 0 0 0 1 1 1
        #     5:  0 0 0 0 0 0 1 1
        #     6:  0 0 0 0 0 0 0 1
        #     7:  0 0 0 0 0 0 0 0
        mask_global = torch.triu(ones, diagonal=1)

        # far_past (遠すぎる過去はマスク: i - j >= sliding_window)
        # sliding_window = 4の場合
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 0 0 0 0 0 0 0
        #     1:  0 0 0 0 0 0 0 0
        #     2:  0 0 0 0 0 0 0 0
        #     3:  0 0 0 0 0 0 0 0
        #     4:  1 0 0 0 0 0 0 0
        #     5:  1 1 0 0 0 0 0 0
        #     6:  1 1 1 0 0 0 0 0
        #     7:  1 1 1 1 0 0 0 0
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T

        # ローカル (スライディングウィンドウ) = 将来 OR 遠過去
        # mask_local
        #     j:  0 1 2 3 4 5 6 7
        # i
        # 0:      0 1 1 1 1 1 1 1
        # 1:      0 0 1 1 1 1 1 1
        # 2:      0 0 0 1 1 1 1 1
        # 3:      0 0 0 0 1 1 1 1
        # 4:      1 0 0 0 0 1 1 1
        # 5:      1 1 0 0 0 0 1 1
        # 6:      1 1 1 0 0 0 0 1
        # 7:      1 1 1 1 0 0 0 0
        mask_local = mask_global | far_past
        return mask_global, mask_local

    def forward(self, input_ids, targets=None):
        """Gemma3モデルの順伝播を実行
        
        Args:
            input_ids (torch.Tensor): 入力トークンID、形状 (batch_size, seq_len)
            targets (torch.Tensor, optional): 目標トークンID（訓練時のみ）、形状 (batch_size, seq_len)
        
        Returns:
            tuple[torch.Tensor, torch.Tensor or None]: ロジットと損失値
                - logits (torch.Tensor): 出力ロジット、形状 (batch_size, seq_len, vocab_size)
                - loss (torch.Tensor or None): 交差エントロピー損失（targetsがNoneの場合はNone）
        
        Behavior:
            1. トークン埋め込みを適用し、スケーリングを実行
            2. アテンションマスクを作成
            3. 各Transformerブロックを順次適用
            4. 最終正規化と出力ヘッドを適用
            5. targetsが指定されている場合、損失を計算
        """
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)
        mask_global, mask_local = self._create_masks(seq_len, x.device)

        for block in self.blocks:
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Gemma3モデルを使用してテキストを生成
        
        Args:
            idx (torch.Tensor): 初期シーケンストークン、形状 (batch_size, seq_len)
            max_new_tokens (int): 生成する新しいトークンの最大数
            temperature (float, optional): サンプリング温度（デフォルト: 1.0）
            top_k (int, optional): Top-Kサンプリングのパラメータ（Noneの場合は無効）
        
        Returns:
            torch.Tensor: 生成されたシーケンス、形状 (batch_size, seq_len + max_new_tokens)
        
        Behavior:
            指定された数の新しいトークンを自動回帰的に生成します。
            コンテキスト長を超える場合は、最新のトークンのみを使用します。
            温度とtop-kサンプリングをサポート
        """
        for _ in range(max_new_tokens):
            ctx_len = self.cfg["context_length"]
            idx_cond = idx if idx.size(1) <= ctx_len else idx[:, -ctx_len:]
            logits, _ = self(idx_cond)  # targets=Noneがデフォルト
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

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

if __name__ == "__main__":

    output_dir = "output"
    learning_rate = 1e-4  # より安定した訓練、以前は1e-4
    max_steps = 10000 # 最大ステップ数(train_loaderの件数より少ない場合のみ、上限として機能)
    warmup_steps = 100
    min_lr = 5e-4
    eval_iters = 200
    gradient_accumulation_steps = 4
    train_batch_size = 1
    vak_batch_size = 1
    max_length = 2048
    stride = 2048
    add_eos_between_documents = True
    eos_token = "<|endoftext|>"
    logging_steps = 1
    weight_decay = 0.1
    betas = (0.9, 0.95)

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(output_dir, exist_ok=True)

    # データセットのロード
    train_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1000]")
    val_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[1000:1100]")

    torch.manual_seed(123)
    model = Gemma3Model(GEMMA3_CONFIG_270M)

    def estimate_loss_with_loader(model, loader, max_batches=None):
        model.eval()
        losses = []
        limit = len(loader) if max_batches is None else min(len(loader), max_batches)
        with torch.inference_mode():
            it = iter(loader)
            for _ in range(limit):
                try:
                    X, Y = next(it)
                except StopIteration:
                    break
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                with ctx:
                    _, loss = model(X, Y)
                losses.append(loss.item())
        model.train()
        return float(np.mean(losses)) if losses else float("inf")

    # トレーニング設定
    import torch
    from contextlib import nullcontext

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # dataloaderの作成（torch.set_default_deviceの前）
    train_loader = create_dataloader_v1(
        train_ds,
        batch_size=train_batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        add_eos_between_documents=add_eos_between_documents,
        eos_token=eos_token,
        pin_memory=(device_type == 'cuda'),
    )

    val_loader = create_dataloader_v1(
        val_ds,
        batch_size=vak_batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        add_eos_between_documents=add_eos_between_documents,
        eos_token=eos_token,
        pin_memory=(device_type == 'cuda'),
    )

    # max_stepsをtrain_loaderの件数に基づいて調整
    # max_stepsがtrain_loaderの件数より少ない場合のみ、上限として機能
    max_steps = min(max_steps, len(train_loader))
    print(f"Training steps: {max_steps} (train_loader size: {len(train_loader)})")

    # autocastの使用方法 https://wandb.ai/wandb_fc/tips/reports/How-To-Use-Autocast-in-PyTorch--VmlldzoyMTk4NTky
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    torch.manual_seed(42)

    from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR

    ## 重み減衰を追加、BETA2を0.95に変更
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay, eps=1e-9)  # 正則化のための重み減衰

    scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)  # 線形ウォームアップの実装
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=min_lr)  # 学習率減衰の実装
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])  # ウォームアップから減衰への切り替え

    # https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

    best_val_loss = float('inf')
    best_model_params_path = os.path.join(output_dir, "best_model_params.pt")
    train_loss_list, validation_loss_list = [], []
    training_log = []
    validation_log = []

    model = model.to(device)

    # 反復可能なイテレータを使って DataLoader から連続的に取り出す
    train_iter = iter(train_loader)
    global_step = 0

    # トレーニングループ（ステップ数で制御）
    pbar = tqdm(total=max_steps)
    while global_step < max_steps:
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)

        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)

        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if ((global_step + 1) % gradient_accumulation_steps == 0) or (global_step + 1 == max_steps):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        global_step += 1
        pbar.update(1)

        # training_logをlogging_stepsごとに記録
        if global_step % logging_steps == 0:
            current_lr = optimizer.param_groups[0]['lr']
            training_log_entry = {
                "step": global_step,
                "train_loss": float(loss.item() * gradient_accumulation_steps),  # 元のloss値
                "learning_rate": current_lr,
                "gradient_accumulation_steps": gradient_accumulation_steps
            }
            training_log.append(training_log_entry)
            with open(os.path.join(output_dir, "training_log.json"), "w") as f:
                json.dump(training_log, f, indent=2)

        # validation_logを評価モード（eval_iters）のときに記録
        if global_step % eval_iters == 0:
            current_lr = optimizer.param_groups[0]['lr']
            train_loss = estimate_loss_with_loader(model, train_loader, max_batches=min(len(train_loader), 50))
            val_loss = estimate_loss_with_loader(model, val_loader, max_batches=min(len(val_loader), 50))
            print(f"Step {global_step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            print(f"The current learning rate: {current_lr:.5f}")

            train_loss_list.append(train_loss)
            validation_loss_list.append(val_loss)

            validation_log_entry = {
                "step": global_step,
                "val_loss": float(val_loss),
            }
            validation_log.append(validation_log_entry)
            with open(os.path.join(output_dir, "validation_log.json"), "w") as f:
                json.dump(validation_log, f, indent=2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

    pbar.close()

    import matplotlib.pyplot as plt
    train_loss_list_converted = list(train_loss_list)
    validation_loss_list_converted = list(validation_loss_list)

    plt.plot(train_loss_list_converted, 'g', label='train_loss')
    plt.plot(validation_loss_list_converted, 'r', label='validation_loss')
    plt.xlabel("Steps - Every 100 epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(output_dir, 'loss_function.png'))

    # モデルをロード
    model = Gemma3Model(GEMMA3_CONFIG_270M)  # 同じ設定でモデルを再作成
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model_params_path = os.path.join(output_dir, "best_model_params.pt")
    model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device)))  # 最良のモデル状態をロード

    sentence = "Neural Networks"
    context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim = 0))
    y = model.generate(context, 200)
    print(enc.decode(y.squeeze().tolist()))