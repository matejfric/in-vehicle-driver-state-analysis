import json
import shutil
from pathlib import Path
from typing import Final

import cv2
from tqdm import tqdm

ROOT: Final[Path] = Path.home() / 'source' / 'driver-dataset' / 'dmd'
CATEGORIES: Final[list[str]] = ['normal', 'anomal']
DISTRACTIONS: Final[list[str]] = [
    'hands_using_wheel/only_right',
    'hands_using_wheel/only_left',
    'driver_actions/radio',
    'driver_actions/drinking',
    'driver_actions/reach_side',
    'driver_actions/unclassified',
]


def setup_output_directories(
    base_dir: str | Path, force_overwrite: bool = False
) -> tuple[Path, Path]:
    """Setup output directories for normal and anomalous/distraction frames."""
    base_dir = Path(base_dir)
    normal_dir = base_dir / 'normal'
    anomaly_dir = base_dir / 'anomal'

    # Clean existing directories if they exist
    for dir_path in [normal_dir, anomaly_dir]:
        if dir_path.exists():
            if not force_overwrite:
                raise ValueError(
                    f'Directory already exists: {dir_path}. Set `force_overwrite=True` to overwrite.'
                )
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True)

    return normal_dir, anomaly_dir


def extract_frames(
    input_video_path: str | Path,
    annotations_file: str | Path,
    distraction_mapping: dict,
    output_base_dir: str | Path,
    target_size: tuple[int, int] | None = None,
    force_overwrite: bool = False,
) -> None:
    """
    Extract frames from video and save them to appropriate directories based on anomaly status.

    Parameters
    ----------
    input_video_path : str | Path
        Path to input video file
    annotations_file : str | Path
        Path to frame annotations in VCD OpenLABEL format (.json)
    distraction_mapping : dict
        Mapping of action keys to anomaly status
    output_base_dir : str | Path
        Base directory for output frames
    target_size : tuple[int, int] | None
        Optional target size for output frames (width, height)

    Example
    -------
    >>> input_video_path = 'gA_1_s1_2019-03-08T09;31;15+01;00_rgb_body.mp4'
    >>> annotations_file = 'gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction.json'
    >>> extract_frames(
            input_video_path=input_video_path,
            annotations_file=annotations_file,
            distraction_mapping=distraction_mapping,
            output_base_dir=input_video_path.replace('.mp4', ''),
            force_overwrite=True,
        )
    """
    # Load annotations
    with open(annotations_file) as f:
        annotations = json.load(f)
    frames_data = annotations['openlabel']['frames']

    # Setup output directories
    normal_dir, anomaly_dir = setup_output_directories(
        output_base_dir, force_overwrite=force_overwrite
    )

    # Initialize video capture
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f'Failed to open video file: {input_video_path}')

    # Track current sequence
    current_sequence = 0
    previous_is_anomaly = None
    current_sequence_dir = None

    # Process frames
    with tqdm(total=len(frames_data)) as pbar:
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame if target size is specified
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

            # Check if current frame has annotations
            frame_key = str(frame_index)
            is_anomaly = False

            frame_annotations = frames_data[frame_key]
            for action_key, _ in frame_annotations['actions'].items():
                if distraction_mapping[int(action_key)] == 1:
                    is_anomaly = True
                    break

            # Create new sequence directory if anomaly status changed
            if previous_is_anomaly is None or is_anomaly != previous_is_anomaly:
                current_sequence += 1
                base_dir = anomaly_dir if is_anomaly else normal_dir
                current_sequence_dir = base_dir / f'sequence_{current_sequence}' / 'rgb'
                current_sequence_dir.mkdir(exist_ok=True, parents=True)
                previous_is_anomaly = is_anomaly

            # Save frame
            if current_sequence_dir is not None:
                pbar.set_postfix_str(f'Sequence: {current_sequence}')
                frame_path = current_sequence_dir / f'{frame_index:06d}.jpg'
                cv2.imwrite(str(frame_path), frame)

            frame_index += 1
            pbar.update(1)

    # Release resources
    cap.release()
