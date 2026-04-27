import torch

from text_summarizer import (
    TransformerSummarizerConfig,
    TransformerTextEncoder,
    TransformerTextSummarizer,
)


def tiny_config() -> TransformerSummarizerConfig:
    return TransformerSummarizerConfig(
        vocab_size=64,
        pad_token_id=0,
        max_source_length=16,
        max_target_length=8,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )


def test_forward_returns_logits_and_loss() -> None:
    model = TransformerTextSummarizer(tiny_config())
    src_tokens = torch.tensor(
        [
            [2, 10, 11, 12, 3, 0],
            [2, 20, 21, 3, 0, 0],
        ]
    )
    tgt_tokens = torch.tensor(
        [
            [1, 30, 31, 3],
            [1, 40, 3, 0],
        ]
    )
    labels = torch.tensor(
        [
            [30, 31, 3, 0],
            [40, 3, 0, 0],
        ]
    )

    output = model(src_tokens=src_tokens, tgt_tokens=tgt_tokens, labels=labels)

    assert output.logits.shape == (2, 4, 64)
    assert output.encoder_last_hidden_state.shape == (2, 6, 32)
    assert output.decoder_last_hidden_state.shape == (2, 4, 32)
    assert output.loss is not None
    assert output.loss.ndim == 0


def test_encoder_returns_hidden_states() -> None:
    encoder = TransformerTextEncoder(
        vocab_size=64,
        pad_token_id=0,
        max_length=16,
        d_model=32,
        num_heads=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )
    src_tokens = torch.tensor([[2, 10, 11, 3, 0]])

    hidden_states = encoder(src_tokens)

    assert hidden_states.shape == (1, 5, 32)


def test_generate_starts_with_bos() -> None:
    model = TransformerTextSummarizer(tiny_config())
    src_tokens = torch.tensor([[2, 10, 11, 3, 0]])

    generated = model.generate(
        src_tokens=src_tokens,
        bos_token_id=1,
        eos_token_id=3,
        max_new_tokens=3,
    )

    assert generated.shape[0] == 1
    assert generated.shape[1] <= 4
    assert generated[0, 0].item() == 1
