"""Strict loading of MLX safetensors shards; never trust the stale BF16 index."""

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten


def load_strict_shards(model, directory, shard_count=3):
    directory = Path(directory)

    filenames = [
        f"model-{i:05d}-of-{shard_count:05d}.safetensors"
        for i in range(1, shard_count + 1)
    ]

    present = {p.name for p in directory.glob("*.safetensors")}
    if present != set(filenames):
        raise ValueError(
            f"Incorrect shard files. "
            f"Missing: {sorted(set(filenames) - present)}; "
            f"Extra: {sorted(present - set(filenames))}"
        )

    weights = {}

    for filename in filenames:
        part = mx.load(str(directory / filename))
        duplicates = weights.keys() & part.keys()

        if duplicates:
            raise ValueError(
                f"Duplicate tensor names: {sorted(duplicates)[:5]}"
            )

        weights.update(part)

    expected = dict(tree_flatten(model.parameters()))

    missing = expected.keys() - weights.keys()
    extra = weights.keys() - expected.keys()

    if missing or extra:
        raise ValueError(
            f"Weight inventory mismatch. "
            f"Missing ({len(missing)}): {sorted(missing)[:5]}; "
            f"Extra ({len(extra)}): {sorted(extra)[:5]}"
        )

    for name, value in weights.items():
        if value.shape != expected[name].shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"checkpoint={value.shape}, "
                f"model={expected[name].shape}"
            )

    model.load_weights(list(weights.items()), strict=True)
    model.eval()

    return len(weights)
