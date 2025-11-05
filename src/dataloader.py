from typing import Iterable, Optional, Union

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(
        self,
        txt: str,
        tokenizer,
        max_length: int,
        stride: int,
    ) -> None:
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Create overlapping windows; if the text is shorter than max_length+1,
        # this loop simply yields no samples (no assertion error).
        for i in range(0, max(0, len(token_ids) - max_length), stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

import tiktoken
import torch


def union_dataset_text(
    source: Union[Iterable[str], "datasets.Dataset"],
    *,
    field: str = "text",
    add_eos_between_documents: bool = True,
    eos_token: str = "<|endoftext|>",
    max_samples: Optional[int] = None,
) -> str:
    """
    HuggingFace Dataset もしくは文字列の反復可能オブジェクトからテキストを結合して
    1 つの長い文字列を返す。各ドキュメントの区切りとして EOS を挿入可能。
    """

    pieces: list[str] = []

    # HuggingFace Dataset の場合（duck typing）
    if hasattr(source, "column_names") and hasattr(source, "__getitem__"):
        total = len(source)
        limit = total if max_samples is None else min(total, max_samples)
        for i in range(limit):
            rec = source[i]
            txt = rec[field] if isinstance(rec, dict) else getattr(rec, field)
            pieces.append(txt)
    else:
        # Iterable[str] を想定
        for idx, txt in enumerate(source):
            if max_samples is not None and idx >= max_samples:
                break
            pieces.append(txt)

    if add_eos_between_documents:
        # ドキュメント境界に EOS を配置（末尾にも 1 つ入る形で OK）
        sep = eos_token
        return sep.join(pieces) + (sep if pieces else "")
    else:
        return "".join(pieces)


def create_dataloader_v1(
    source: Union[str, Iterable[str], "datasets.Dataset"],
    *,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    field: str = "text",
    add_eos_between_documents: bool = True,
    eos_token: str = "<|endoftext|>",
    pin_memory: bool = False,
) -> DataLoader:
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Prepare text
    if isinstance(source, str):
        txt = source
    else:
        txt = union_dataset_text(
            source,
            field=field,
            add_eos_between_documents=add_eos_between_documents,
            eos_token=eos_token,
        )

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader


if __name__ == "__main__":
    
    train_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1000]")
    val_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[1000:1100]")

    tokenized_train_ds = create_dataloader_v1(
        train_ds,
        batch_size=4,
        max_length=4096,
        stride=4096,
        shuffle=False,
        add_eos_between_documents=True,
        eos_token="<|endoftext|>",
    )
    print(tokenized_train_ds)
    print(iter(tokenized_train_ds))
    print(len(tokenized_train_ds))
    print(next(iter(tokenized_train_ds)))
    
    tokenized_val_ds = create_dataloader_v1(
        val_ds,
        batch_size=4,
        max_length=4096,
        stride=4096,
        shuffle=False,
        add_eos_between_documents=True,
        eos_token="<|endoftext|>",
    )
    print(tokenized_val_ds)
    print(iter(tokenized_val_ds))
    print(len(tokenized_val_ds))
    print(next(iter(tokenized_val_ds)))