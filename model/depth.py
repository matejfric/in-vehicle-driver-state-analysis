import logging
from pathlib import Path
from pprint import pprint
from typing import Literal

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
    source_video_extension: str = 'mp4',
    checkpoint_dir: str | Path = 'model/depth_anything/checkpoints',
    input_size: int = 518,
    use_parent_dir: bool = False,
    save_visualization: bool = True,
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

    all_sequences = sorted([p for p in base_directory.rglob(f'*/{source_type}')])
    pprint(all_sequences)

    model_config = {
        'encoder': encoder,
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
        videos = sorted(seq_dir.glob(f'*.{source_video_extension}'))
        if not videos:
            logger.warning(f'No videos found in {seq_dir}')
            continue

        current_frame = 0
        for i, video in enumerate(videos):
            pbar.set_postfix_str(str(video))
            frames, target_fps = read_video_frames(str(seq_dir / video), -1, -1, -1)
            depths, fps = video_depth_anything.infer_video_depth(
                frames, target_fps, device=device, input_size=input_size
            )
            output_dir = (
                seq_dir.parent.parent if use_parent_dir else seq_dir.parent
            ) / 'video_depth_anything'
            _export_depth_frames(
                output_dir,
                depths,
                filenames=sorted(
                    (seq_dir.parent if use_parent_dir else seq_dir).glob(
                        f'*.{source_extension}'
                    )
                )[current_frame : current_frame + len(depths)],
            )
            if save_visualization:
                visu_name = 'video_depth_anything'
                output_file_name = (
                    f'{visu_name}_{i:03d}.mp4'
                    if len(videos) > 1
                    else f'{visu_name}.mp4'
                )
                save_video(
                    depths,
                    str(output_dir / output_file_name),
                    fps=fps,
                    is_depths=True,
                )
            current_frame += len(depths)
