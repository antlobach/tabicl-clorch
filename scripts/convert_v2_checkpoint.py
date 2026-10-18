#!/usr/bin/env python3
"""Convert an official TabICLv2 PyTorch checkpoint to SafeTensors + JSON."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict")
    config = checkpoint.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError("Expected an official TabICLv2 checkpoint with state_dict and config")

    tensors = {name: value.detach().contiguous() for name, value in state.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    args.output.with_suffix(".json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(tensors)} tensors to {args.output}")


if __name__ == "__main__":
    main()
