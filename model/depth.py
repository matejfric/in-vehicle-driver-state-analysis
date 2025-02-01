from pathlib import Path
from pprint import pprint
from typing import Literal

import numpy as np
from PIL import Image
from tqdm import tqdm


def _export_depth_frames(
    output_dir: str | Path, depth_frames: np.ndarray, filenames: list[Path]
) -> None:
    """Export depth frames to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Normalize to 0-255 range
    d_min = depth_frames.min()
    d_max = depth_frames.max()
    depth_frames = (depth_frames - d_min) / (d_max - d_min) * 255
    depth_frames = depth_frames.astype(np.uint8)

    # Save each frame as a PNG
    for i in range(depth_frames.shape[0]):
        img = Image.fromarray(depth_frames[i])
        img.save(output_dir / f'{filenames[i].stem}.png')


def convert_video_to_depth(
    base_directory: str | Path,
    encoder: Literal['vits', 'vitl'] = 'vits',
    source_type: str = 'crop_rgb',
    source_extension: str = 'jpg',
    checkpoint_dir: str | Path = 'model/depth_anything/checkpoints',
) -> None:
    """Convert RGB videos to depth maps using Video-Depth-Anything model."""
    import torch

    from .depth_anything.utils.dc_utils import read_video_frames, save_video
    from .depth_anything.video_depth_anything.video_depth import (
        VideoDepthAnything,
    )

    base_directory = Path(base_directory)
    if not base_directory.exists():
        raise ValueError(f'Session directory not found: {base_directory}')

    all_sequences = sorted([p for p in base_directory.rglob(source_type) if p.is_dir()])
    pprint(all_sequences)

    model_config = {
        'encoder': 'vits',
        'features': 64,
        'out_channels': [48, 96, 192, 384],
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_depth_anything = VideoDepthAnything(**model_config)
    video_depth_anything.load_state_dict(
        torch.load(
            Path(checkpoint_dir) / f'video_depth_anything_{encoder}.pth',
            map_location='cpu',
        ),
        strict=True,
    )
    video_depth_anything = video_depth_anything.to(device).eval()

    for seq_dir in (pbar := tqdm(all_sequences)):
        pbar.set_postfix_str(seq_dir.parent.name)

        frames, target_fps = read_video_frames(str(seq_dir / 'video.mp4'), -1, -1, -1)
        depths, fps = video_depth_anything.infer_video_depth(
            frames, target_fps, device=device
        )
        output_dir = seq_dir.parent / 'video_depth_anything'
        _export_depth_frames(
            output_dir, depths, filenames=sorted(seq_dir.glob(f'*.{source_extension}'))
        )
        save_video(
            depths,
            str(output_dir / 'video_depth_anything.mp4'),
            fps=fps,
            is_depths=True,
        )
