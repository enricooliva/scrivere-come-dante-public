"""Addestra o usa un piccolo Transformer decoder-only sulla Divina Commedia."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    batch_size: int = 64
    block_size: int = 256
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    eval_iters: int = 200
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    seed: int = 1337


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/commedia.txt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gpt_commedia.pt"))
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--samples", type=Path, default=Path("results/samples.txt"))
    parser.add_argument("--prompt", default="Nel mezzo del cammin")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Config:
    if not path.exists():
        return Config()
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    return Config(**values)


class Head(nn.Module):
    def __init__(self, config: Config, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time_steps, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        weights = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        weights = weights.masked_fill(self.tril[:time_steps, :time_steps] == 0, float("-inf"))
        weights = self.dropout(F.softmax(weights, dim=-1))
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd deve essere divisibile per n_head")
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(torch.cat([head(x) for head in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm_1 = nn.LayerNorm(config.n_embd)
        self.norm_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_1(x))
        return x + self.feed_forward(self.norm_2(x))


class GPTLanguageModel(nn.Module):
    def __init__(self, config: Config, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        batch, time_steps = idx.shape
        token = self.token_embedding(idx)
        position = self.position_embedding(torch.arange(time_steps, device=idx.device))
        logits = self.output(self.final_norm(self.blocks(token + position)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(batch * time_steps, -1), targets.reshape(batch * time_steps))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature deve essere maggiore di zero")
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def build_vocabulary(text: str):
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for index, char in enumerate(chars)}

    def encode(value: str) -> list[int]:
        missing = sorted(set(value) - set(stoi))
        if missing:
            raise ValueError(f"Il prompt contiene caratteri fuori vocabolario: {missing}")
        return [stoi[char] for char in value]

    def decode(values: list[int]) -> str:
        return "".join(itos[index] for index in values)

    return chars, encode, decode


def save_checkpoint(path: Path, model: GPTLanguageModel, chars: list[str], config: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "chars": chars, "config": config.__dict__},
        path,
    )


def remap_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    renamed: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("token_embedding_table."):
            new_key = new_key.replace("token_embedding_table.", "token_embedding.", 1)
        elif new_key.startswith("position_embedding_table."):
            new_key = new_key.replace("position_embedding_table.", "position_embedding.", 1)
        elif new_key.startswith("ln_f."):
            new_key = new_key.replace("ln_f.", "final_norm.", 1)
        elif new_key.startswith("lm_head."):
            new_key = new_key.replace("lm_head.", "output.", 1)

        new_key = new_key.replace(".sa.", ".attention.")
        new_key = new_key.replace(".ffwd.", ".feed_forward.")
        new_key = new_key.replace(".ln1.", ".norm_1.")
        new_key = new_key.replace(".ln2.", ".norm_2.")
        renamed[new_key] = value
    return renamed


def load_checkpoint(path: Path, device: str, chars: list[str]) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("chars") != chars:
        raise ValueError("Il vocabolario del checkpoint non coincide con quello del corpus")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Il checkpoint non contiene un model_state_dict valido")

    checkpoint["model_state_dict"] = remap_legacy_state_dict(state_dict)
    return checkpoint


def append_metrics(path: Path, step: int, train_loss: float, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["step", "train_loss", "val_loss"])
        writer.writerow([step, f"{train_loss:.6f}", f"{val_loss:.6f}"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.data.exists():
        raise FileNotFoundError(f"Corpus non trovato: {args.data}")
    text = args.data.read_text(encoding="utf-8")
    chars, encode, decode = build_vocabulary(text)
    data = torch.tensor(encode(text), dtype=torch.long)
    split_index = int(0.9 * len(data))
    train_data, val_data = data[:split_index], data[split_index:]

    def get_batch(split: str):
        source = train_data if split == "train" else val_data
        if len(source) <= config.block_size:
            raise ValueError("Il corpus è troppo corto rispetto alla finestra di contesto")
        starts = torch.randint(len(source) - config.block_size, (config.batch_size,))
        x = torch.stack([source[i : i + config.block_size] for i in starts])
        y = torch.stack([source[i + 1 : i + config.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    model = GPTLanguageModel(config, len(chars)).to(device)
    print(f"Dispositivo: {device}")
    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")

    if args.checkpoint.exists():
        checkpoint = load_checkpoint(args.checkpoint, device, chars)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint caricato: {args.checkpoint}")

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        model.eval()
        result: dict[str, float] = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(config.eval_iters):
                inputs, targets = get_batch(split)
                _, loss = model(inputs, targets)
                assert loss is not None
                losses.append(loss.item())
            result[split] = sum(losses) / len(losses)
        model.train()
        return result

    def generate_text(prompt: str, tokens: int, temperature: float) -> str:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        generated = model.generate(context, tokens, temperature=temperature)[0].tolist()
        return decode(generated)

    if not args.skip_train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        args.samples.parent.mkdir(parents=True, exist_ok=True)
        for step in range(config.max_iters):
            if step % config.eval_interval == 0 or step == config.max_iters - 1:
                losses = estimate_loss()
                print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
                append_metrics(args.metrics, step, losses["train"], losses["val"])
                sample = generate_text(args.prompt, min(args.tokens, 300), args.temperature)
                with args.samples.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n\n=== step {step} ===\n{sample}\n")

            inputs, targets = get_batch("train")
            _, loss = model(inputs, targets)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        save_checkpoint(args.checkpoint, model, chars, config)
        print(f"Checkpoint salvato: {args.checkpoint}")
    elif not args.checkpoint.exists():
        raise FileNotFoundError("Con --skip-train è necessario un checkpoint esistente")

    print(generate_text(args.prompt, args.tokens, args.temperature))


if __name__ == "__main__":
    main()
