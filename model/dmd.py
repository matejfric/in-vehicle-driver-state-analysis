import json
import shutil
from functools import cache
from pathlib import Path
from typing import Final, Literal

import cv2
from tqdm import tqdm

ROOT: Final[Path] = Path.home() / 'source' / 'driver-dataset' / 'dmd'
CATEGORIES: Final[list[str]] = ['normal', 'anomal']
DISTRACTIONS: Final[list[str]] = [
    'hands_using_wheel/only_right',
    'hands_using_wheel/only_left',
    'hands_using_wheel/none',
    'driver_actions/radio',
    'driver_actions/drinking',
    'driver_actions/reach_side',
    'driver_actions/unclassified',
    'driver_actions/hair_and_makeup',
    'driver_actions/phonecall_right',
    'driver_actions/phonecall_left',
    'driver_actions/texting_left',
    'driver_actions/texting_right',
]
OTHER_ACTIONS: Final[list[str]] = [
    'gaze_on_road/looking_road',
    'gaze_on_road/not_looking_road',
    'hands_using_wheel/both',
    'talking/talking',
    'driver_actions/talking_to_passenger',
    'driver_actions/safe_drive',
]
SOURCE_TYPES: Final[list[str]] = ['rgb', 'depth']


@cache
def _load_annotations(annotations_file_path: str | Path) -> dict:
    with open(annotations_file_path) as f:
        return json.load(f)


def _setup_output_directories(
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


def _get_action_mapping(annotations_file_path: str | Path) -> dict[int, str]:
    """Get action mapping from annotations file."""
    annotations = _load_annotations(annotations_file_path)
    actions = annotations['openlabel']['actions']
    return {int(k): v['type'] for k, v in actions.items()}


def _get_distraction_mapping(annotations_file_path: str | Path) -> dict[int, int]:
    """Get distraction mapping from annotations file."""
    actions = _get_action_mapping(annotations_file_path)
    return {k: 1 if v in DISTRACTIONS else 0 for k, v in actions.items()}


def extract_frames(
    input_video_path: str | Path,
    annotations_file_path: str | Path,
    source_type: Literal['rgb', 'depth'],
    output_base_dir: str | Path | None = None,
    target_size: tuple[int, int] | None = None,
    force_overwrite: bool = False,
) -> None:
    """
    Extract frames from video and save them to appropriate directories based on anomaly status.

    Parameters
    ----------
    input_video_path : str | Path
        Path to input video file
    annotations_file_path : str | Path
        Path to frame annotations in VCD OpenLABEL format (.json)
    output_base_dir : str | Path | None
        Base directory for output frames
    target_size : tuple[int, int] | None
        Optional target size for output frames (width, height)
    force_overwrite : bool
        Whether to overwrite existing output directories

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
    if source_type not in SOURCE_TYPES:
        raise ValueError(f'Invalid source type: {source_type}')

    if output_base_dir is None:
        output_base_dir = Path(input_video_path).parent

    # Load annotations
    annotations = _load_annotations(annotations_file_path)
    frames_data = annotations['openlabel']['frames']

    actions = _get_action_mapping(annotations_file_path)
    known_actions = DISTRACTIONS + OTHER_ACTIONS
    assert all(action in known_actions for action in actions.values()), (
        f'Unknown actions found in annotations file: {actions.values()}'
    )

    distraction_mapping = _get_distraction_mapping(annotations_file_path)

    # Setup output directories
    normal_dir, anomaly_dir = _setup_output_directories(
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

            frame_key = str(frame_index)
            is_anomaly = False
            frame_annotations = frames_data[frame_key]
            for action_key, _ in frame_annotations['actions'].items():
                if distraction_mapping[int(action_key)] == 1:
                    # One of the annotated actions for the frame is a distraction
                    is_anomaly = True
                    break

            # Create new sequence directory if anomaly status changed
            if previous_is_anomaly is None or is_anomaly != previous_is_anomaly:
                current_sequence += 1
                base_dir = anomaly_dir if is_anomaly else normal_dir
                current_sequence_dir = (
                    base_dir / f'sequence_{current_sequence}' / source_type
                )
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
