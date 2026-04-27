from __future__ import annotations

from pathlib import Path

import torch

from .model import TransformerTextSummarizer, load_model_bundle
from .tokenizer import SimpleTokenizer


def load_for_inference(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[TransformerTextSummarizer, SimpleTokenizer, dict[str, list[float]]]:
    model, tokenizer, history = load_model_bundle(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model, tokenizer, history


@torch.no_grad()
def summarize(
    model: TransformerTextSummarizer,
    tokenizer: SimpleTokenizer,
    text: str,
    device: torch.device,
    max_source_length: int | None = None,
    max_new_tokens: int | None = None,
) -> str:
    max_source_length = max_source_length or model.config.max_source_length
    max_new_tokens = max_new_tokens or model.config.max_target_length
    src_tokens = tokenizer.encode_source(text, max_source_length)
    src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)
    generated = model.generate(
        src_tensor,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.decode(generated[0].detach().cpu().tolist())

