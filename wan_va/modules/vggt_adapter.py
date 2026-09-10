import os
import sys
import math
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except Exception as e:
    pass

import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Tuple
from einops import rearrange

from wan_va.utils.logging import logger, init_logger
from wan_va.modules.visual_util import predictions_to_glb
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """Convert depth maps to 3D world points."""
    depth = depth_map[..., 0]
    leading_shape = depth.shape[:-2]
    height, width = depth.shape[-2:]
    num_frames = int(np.prod(leading_shape))
    depth = depth.reshape(num_frames, height, width)
    extrinsic = extrinsic.reshape(num_frames, *extrinsic.shape[-2:])
    intrinsic = intrinsic.reshape(num_frames, *intrinsic.shape[-2:])

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    ).reshape(*leading_shape, height, width, 3)


def save_glb(
    predictions: dict,
    output_dir: str,
    conf_thres: float = 40.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    max_points: int = 1000000,
) -> str:
    """Generate and save GLB visualization."""
    if predictions_to_glb is None:
        raise ImportError("predictions_to_glb is unavailable because fastwam.utils.visual_util is not installed")

    conf_thres = max(3.0, float(conf_thres))

    glb_path = os.path.join(
        output_dir,
        f"scene_conf{conf_thres}_black{mask_black_bg}_white{mask_white_bg}_"
        f"cam{show_cam}_sky{mask_sky}_max{max_points // 1000}k.glb",
    )

    scene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=output_dir,
        max_points=max_points,
    )
    scene.export(glb_path)
    return glb_path


class VGGTAdapter(nn.Module):
    """
    VGGTAdapter integrates VGGTOmega models.

    This adapter provides:
    - encode: Transform images to latent representations via VGGTOmega aggregator
    - decode: Transform latent representations back to 3D geometry predictions
    - forward: End-to-end pipeline (encode + decode)
    """

    def __init__(
        self,
        vggt: Optional[VGGTOmega] = None,
        default_image_size: Optional[list] = None,
        latent_frame_mode: Optional[str] = None,
        latent_dimension: int = 2048,
    ):
        super().__init__()
        self.vggt = vggt
        self.default_image_size = default_image_size
        self.latent_frame_mode = latent_frame_mode

    @classmethod
    def from_pretrained(
        cls,
        vggt_adapter_config,
        vggt_pretrained_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ) -> "VGGTAdapter":
        """
        Load VGGTAdapter from pretrained checkpoints.

        Args:
            vggt_adapter_config: Config of VGGTOmega and adapter
            vggt_pretrained_path: Path to VGGTOmega checkpoint directory
            device: Target device (cuda/cpu)
            torch_dtype: Target dtype (default: bfloat16)

        Returns:
            VGGTAdapter instance with loaded models
        """
        # Load VGGT
        if not os.path.isfile(vggt_pretrained_path):
            raise FileNotFoundError(f"VGGTOmega checkpoint not found at {vggt_pretrained_path}")
        logger.info(f"Loading VGGTOmega from {vggt_pretrained_path}...")
        vggt = VGGTOmega()
        state_dict = torch.load(vggt_pretrained_path, map_location="cpu")
        vggt.load_state_dict(state_dict)

        return cls(
            vggt=vggt.to(device, torch_dtype),
            default_image_size=vggt_adapter_config["default_image_size"],
            latent_frame_mode=vggt_adapter_config["latent_frame_mode"],
            latent_dimension=vggt_adapter_config["latent_dimension"]
        )

    def process_images(
        self,
        images: Optional[np.ndarray | torch.Tensor] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Encode images to latent representations.

        Args:
            images: Input images [B, C, F, H, W] or [C, F, H, W], in range [0, 1]
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        if len(images.shape) == 4:
            images = images.unsqueeze(0) # [B, C, F, H, W]

        # Rearrage images
        batch_size = len(images)
        images = rearrange(images, 'b c f h w -> (b f) c h w')
        images = load_and_preprocess_images(images, image_resolution=self.default_image_size[0])
        images = rearrange(images, '(b f) c h w -> b f c h w', b=batch_size).to(device, torch_dtype)
        return images

    def encode(
        self,
        images: Optional[np.ndarray | torch.Tensor] = None,
        return_dict: bool = True,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ) -> Dict[str, torch.Tensor]:
        """
        Encode images to latent representations.

        Args:
            images: Input images [B, C, F, H, W] or [C, F, H, W], in range [0, 1]
            return_dict: Whether to return a dictionary
        """
        if self.vggt is None:
            raise RuntimeError("VGGTOmega model not loaded. Call from_pretrained() first.")

        # Process images
        images = self.process_images(images, device, torch_dtype) # return [B, F, C, H, W]

        # Get aggregated tokens from VGGT
        device_str = device.type if isinstance(device, torch.device) else device
        with torch.inference_mode():
            with torch.autocast(device_type=device_str, dtype=torch_dtype):
                aggregated_tokens_list, patch_token_start = self.vggt.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")
        latents = final_tokens[:, :, :patch_token_start].contiguous().to(torch_dtype) # [B, F, 17, 2048]

        if not return_dict:
            return latents, images, aggregated_tokens_list, patch_token_start

        return {
            "latents": latents,
            "images": images,
            "aggregated_tokens_list": aggregated_tokens_list,
            "patch_token_start": patch_token_start
        }

    def decode(
        self,
        images: torch.Tensor = None,
        aggregated_tokens_list: torch.Tensor = None,
        patch_token_start: int = None,
        device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent representations to 3D geometry predictions.

        Args:
            images: Processed images tensor for pose encoding
            aggregated_tokens_list: Predictions from encode
            patch_token_start: Patch from start
            return_dict: Whether to return a dictionary
        """
        if self.vggt is None:
            raise RuntimeError("VGGTOmega model not loaded. Call from_pretrained() first.")

        predictions = {
            "images": images
        }

        # run heads
        device_str = device.type if isinstance(device, torch.device) else device
        with torch.inference_mode():
            with torch.autocast(device_type=device_str, enabled=False):
                if self.vggt.camera_head is not None:
                    predictions["pose_enc"] = self.vggt.camera_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )

                if self.vggt.dense_head is not None:
                    depth, depth_conf = self.vggt.dense_head(
                        aggregated_tokens_list,
                        images=images,
                        patch_token_start=patch_token_start,
                    )
                    predictions["depth"] = depth
                    predictions["depth_conf"] = depth_conf

        # postprocess
        if "pose_enc" in predictions:
            extrinsic, intrinsic = encoding_to_camera(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],
            )
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic

        predictions_np = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().cpu().numpy()
                if value.shape[0] == 1:
                    value = value[0]
                predictions_np[key] = value

        if "depth" in predictions_np and "extrinsic" in predictions_np:
            predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
                predictions_np["depth"],
                predictions_np["extrinsic"],
                predictions_np["intrinsic"],
            )
        return predictions_np

    def forward(
        self,
        images: Optional[np.ndarray | torch.Tensor] = None,
        return_latents: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode images and decode to 3D geometry predictions.

        Args:
            images: Input images [B, C, F, H, W] or [C, F, H, W], in range [0, 1]
            return_latent: Whether to include latent in output
        """
        # Encode
        encode_output = self.encode(images, return_dict=True, device=device, torch_dtype=torch_dtype)
        # Decode
        decode_output = self.decode(encode_output["images"], encode_output["aggregated_tokens_list"],
                                    encode_output["patch_token_start"], device)

        # Combine results
        if return_latents:
            decode_output["latents"] = encode_output["latents"].float().cpu().numpy()

        return decode_output


def main():
    """
    Test script for VGGTAdapter forward functions.
    ```bash
        cd /efs-mi-east4-2/lijianwen/Codes/FastWAM
        python -m fastwam.models.wan22.vggt_adapter
    ```
    """
    import argparse
    import glob
    from PIL import Image

    parser = argparse.ArgumentParser(description="Test VGGTAdapter encode/decode/forward")
    parser.add_argument("--vggt_path", type=str, default="/efs-mi-east4-2/lijianwen/Pretrained_models/VGGT/VGGT-Omega/vggt_omega_1b_512.pt", help="Path to VGGT checkpoint")
    parser.add_argument("--image_dir", type=str, default="/efs-mi-east4-2/lijianwen/Codes/lingbot-va/example/robotwin", help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="./vggt_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--default_image_size", type=int, nargs=2, default=[512, 512], help="Default image size [H, W]")
    parser.add_argument("--latent_frame_mode", type=str, default="concat", help="Default frame mode")
    parser.add_argument("--latent_dimension", type=int, default=2048, help="Default frame dimension")
    args = parser.parse_args()

    # Determine dtype
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    # Create config
    vggt_adapter_config = {
        "default_image_size": args.default_image_size,
        "latent_frame_mode": args.latent_frame_mode,
        "latent_dimension": args.latent_dimension
    }

    # Load model
    print(f"Loading VGGTAdapter...")
    print(f"  VGGTOmega path: {args.vggt_path}")
    print(f"  Device: {args.device}, dtype: {torch_dtype}")

    adapter = VGGTAdapter.from_pretrained(
        vggt_adapter_config=vggt_adapter_config,
        vggt_pretrained_path=args.vggt_path,
        device=args.device,
        torch_dtype=torch_dtype,
    )
    adapter.eval()

    # Load images from directory
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {args.image_dir}")

    print(f"Found {len(image_paths)} images:")
    for p in image_paths:
        print(f"  {p}")

    # Load images using PIL
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)

    # Stack images: [F, H, W, C] -> [C, F, H, W]
    images = np.stack(images, axis=0)  # [F, H, W, C]
    images = images.transpose(3, 0, 1, 2)  # [C, F, H, W]
    print(f"Input images shape: {images.shape}")

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        forward_outputs = adapter(images, return_latents=True, device=args.device, torch_dtype=torch_dtype)
        print(f"Forward output keys: {list(forward_outputs.keys())}")
        print(f"  depth shape: {forward_outputs['depth'].shape}")
        print(f"  world_points_from_depth shape: {forward_outputs['world_points_from_depth'].shape}")
        print(f"  extrinsic shape: {forward_outputs['extrinsic'].shape}")
        print(f"  intrinsic shape: {forward_outputs['intrinsic'].shape}")
        print(f"  images shape: {forward_outputs['images'].shape}")
        print(f"  latents shape: {forward_outputs['latents'].shape}")

        glb_path = save_glb(forward_outputs, args.output_dir)
        print(f"  Saved GLB to {glb_path}")


if __name__ == "__main__":
    init_logger()
    main()
