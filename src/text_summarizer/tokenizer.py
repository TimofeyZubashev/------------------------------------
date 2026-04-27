from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


class SimpleTokenizer:
    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, min_freq: int = 2, max_vocab_size: int = 30000) -> None:
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.token_to_id = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = text.lower().strip()
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def fit(self, texts: list[str]) -> None:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self.tokenize(text))

        max_new_tokens = self.max_vocab_size - len(self.token_to_id)
        for token, count in counts.most_common(max_new_tokens):
            if count < self.min_freq:
                continue
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def ids(self, text: str) -> list[int]:
        return [
            self.token_to_id.get(token, self.unk_token_id)
            for token in self.tokenize(text)
        ]

    def pad(self, token_ids: list[int], max_length: int) -> list[int]:
        token_ids = token_ids[:max_length]
        return token_ids + [self.pad_token_id] * (max_length - len(token_ids))

    def encode_source(self, text: str, max_length: int) -> list[int]:
        token_ids = self.ids(text)[: max_length - 1] + [self.eos_token_id]
        return self.pad(token_ids, max_length)

    def encode_target_pair(
        self,
        text: str,
        max_length: int,
    ) -> tuple[list[int], list[int]]:
        token_ids = self.ids(text)[: max_length - 1]
        decoder_input = [self.bos_token_id] + token_ids
        labels = token_ids + [self.eos_token_id]
        return self.pad(decoder_input, max_length), self.pad(labels, max_length)

    def decode(self, token_ids: list[int]) -> str:
        skip = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens = [
            self.id_to_token.get(int(idx), self.unk_token)
            for idx in token_ids
            if int(idx) not in skip
        ]
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text.strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
            "token_to_id": self.token_to_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimpleTokenizer":
        tokenizer = cls(
            min_freq=int(data.get("min_freq", 2)),
            max_vocab_size=int(data.get("max_vocab_size", 30000)),
        )
        tokenizer.token_to_id = {
            str(token): int(idx)
            for token, idx in data["token_to_id"].items()
        }
        tokenizer.id_to_token = {
            int(idx): token
            for token, idx in tokenizer.token_to_id.items()
        }
        return tokenizer

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

