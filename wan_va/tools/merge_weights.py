#!/usr/bin/env python
"""Merge a base Wan transformer checkpoint with newly added VGGT modules."""

import sys
import argparse
import inspect
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEFAULT_SOURCE = Path("/efs-mi-east4-2/lijianwen/Pretrained_models/lingbot-va/lingbot-va-base/transformer")
DEFAULT_OUTPUT = Path("/efs-mi-east4-2/lijianwen/Pretrained_models/lingbot-va/lingbot-vggt-base/transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a complete WanTransformer3DModel checkpoint by loading the "
            "compatible tensors from an existing transformer checkpoint and "
            "keeping default initialization for newly added modules."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    return parser.parse_args()


def load_config(source_dir: Path) -> dict:
    config_path = source_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with config_path.open() as f:
        return json.load(f)


def model_kwargs_from_config(config: Mapping[str, object]) -> dict:
    from wan_va.modules.model import WanTransformer3DModel

    signature = inspect.signature(WanTransformer3DModel.__init__)
    allowed = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in config.items() if key in allowed}


def build_initialized_model(config: Mapping[str, object], dtype: torch.dtype | None, seed: int):
    from wan_va.modules.model import WanTransformer3DModel

    torch.manual_seed(seed)
    model = WanTransformer3DModel(**model_kwargs_from_config(config))
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    return model


def discover_safetensor_files(source_dir: Path) -> dict[str, Path]:
    index_path = source_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            index = json.load(f)
        return {
            key: source_dir / filename
            for key, filename in index.get("weight_map", {}).items()
        }

    tensor_files = sorted(source_dir.glob("*.safetensors"))
    if not tensor_files:
        raise FileNotFoundError(f"No safetensors files found in {source_dir}")

    key_to_file: dict[str, Path] = {}
    for tensor_file in tensor_files:
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                key_to_file[key] = tensor_file
    return key_to_file


def infer_checkpoint_dtype(source_dir: Path) -> torch.dtype | None:
    key_to_file = discover_safetensor_files(source_dir)
    for tensor_file in sorted(set(key_to_file.values())):
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if tensor.is_floating_point():
                    return tensor.dtype
    return None


def copy_compatible_tensor(
    target_state: Mapping[str, torch.Tensor],
    target_key: str,
    source_state: Mapping[str, torch.Tensor],
) -> str:
    if target_key not in source_state:
        raise KeyError(f"Missing required checkpoint key: {target_key}")

    source_tensor = source_state[target_key]
    target_tensor = target_state[target_key]
    if tuple(source_tensor.shape) != tuple(target_tensor.shape):
        raise ValueError(
            "Checkpoint tensor shape mismatch: "
            f"{target_key} {tuple(source_tensor.shape)} -> "
            f"{target_key} {tuple(target_tensor.shape)}"
        )

    target_tensor.copy_(source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype))
    return "loaded"


def plan_source_to_targets(
    target_keys: set[str],
    source_keys: set[str],
) -> dict[str, list[str]]:
    source_to_targets: dict[str, list[str]] = defaultdict(list)
    for target_key in sorted(target_keys):
        if target_key in source_keys:
            source_to_targets[target_key].append(target_key)
    return source_to_targets


def merge_checkpoint(model, source_dir: Path) -> Counter:
    target_state = model.state_dict()
    key_to_file = discover_safetensor_files(source_dir)
    source_to_targets = plan_source_to_targets(
        set(target_state.keys()), set(key_to_file.keys())
    )
    missing_keys = sorted(set(target_state.keys()) - set(source_to_targets.keys()))
    expected_new_keys = {
        "vggt_pre_norm.weight",
        "vggt_pre_norm.bias",
        "vggt_patch_embedding_mlp.weight",
        "vggt_patch_embedding_mlp.bias",
        "vggt_proj_out.weight",
        "vggt_proj_out.bias",
    }
    unexpected_missing_keys = [
        key for key in missing_keys if key not in expected_new_keys
    ]
    if unexpected_missing_keys:
        preview = ", ".join(unexpected_missing_keys[:20])
        raise KeyError(
            "Missing required checkpoint keys: "
            f"{preview}"
            + (" ..." if len(unexpected_missing_keys) > 20 else "")
        )

    counters = Counter()
    used_source_keys: set[str] = set()

    targets_by_file: dict[Path, list[str]] = defaultdict(list)
    for source_key in source_to_targets:
        targets_by_file[key_to_file[source_key]].append(source_key)

    with torch.no_grad():
        for tensor_file in sorted(targets_by_file):
            with safe_open(tensor_file, framework="pt", device="cpu") as handle:
                for source_key in sorted(targets_by_file[tensor_file]):
                    source_tensor = handle.get_tensor(source_key)
                    used_source_keys.add(source_key)
                    for target_key in source_to_targets[source_key]:
                        target_tensor = target_state[target_key]
                        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                            raise ValueError(
                                "Checkpoint tensor shape mismatch: "
                                f"{source_key} {tuple(source_tensor.shape)} -> "
                                f"{target_key} {tuple(target_tensor.shape)}"
                            )
                        target_tensor.copy_(
                            source_tensor.to(
                                device=target_tensor.device,
                                dtype=target_tensor.dtype,
                            )
                        )
                        counters["loaded"] += 1

    counters["initialized_new"] = len(missing_keys)
    counters["unexpected"] = len(set(key_to_file.keys()) - used_source_keys)
    return counters


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    output_dir = args.output.resolve()

    config = load_config(source_dir)
    dtype = infer_checkpoint_dtype(source_dir)
    model = build_initialized_model(config, dtype=dtype, seed=args.seed)
    counters = merge_checkpoint(model, source_dir)

    prepare_output_dir(output_dir, overwrite=args.overwrite)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    print(f"source: {source_dir}")
    print(f"output: {output_dir}")
    print(f"dtype: {dtype}")
    print(f"seed: {args.seed}")
    print(f"loaded exact tensors: {counters['loaded']}")
    print(f"kept initialized new tensors: {counters['initialized_new']}")
    print(f"unused source tensors: {counters['unexpected']}")


if __name__ == "__main__":
    main()
