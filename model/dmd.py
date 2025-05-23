import json
import shutil
from functools import cache
from pathlib import Path
from pprint import pprint
from typing import Final, Literal

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from model.common import crop_aspect_ratio, pad_to_square, preprocess_depth_frame

ROOT: Final[Path] = Path.home() / 'source' / 'driver-dataset' / 'dmd'
CATEGORIES: Final[list[str]] = ['normal', 'anomal']
DISTRACTIONS: Final[list[str]] = [
    'hands_using_wheel/only_right',
    'hands_using_wheel/only_left',
    'hands_using_wheel/none',
    'hands_on_wheel/only_left',
    'hands_on_wheel/only_right',
    'hands_on_wheel/none',
    'driver_actions/radio',
    'driver_actions/drinking',
    'driver_actions/reach_side',
    'driver_actions/unclassified',
    'driver_actions/hair_and_makeup',
    'driver_actions/phonecall_right',
    'driver_actions/phonecall_left',
    'driver_actions/texting_left',
    'driver_actions/texting_right',
    'driver_actions/reach_backseat',
    'yawning/Yawning with hand',
]
DROWSINESS: Final[list[str]] = [
    'blinks/blinking',
    'eyes_state/opening',
    'eyes_state/close',
    'eyes_state/open',
    'eyes_state/closing',
    'eyes_state/undefined',
    'yawning/Yawning without hand',
]
OTHER_ACTIONS: Final[list[str]] = DROWSINESS + [
    'gaze_on_road/looking_road',
    'gaze_on_road/not_looking_road',
    'hands_using_wheel/both',
    'hands_on_wheel/both_hands',
    'talking/talking',
    'driver_actions/standstill_or_waiting',  # This may be classified as a distraction
    'driver_actions/talking_to_passenger',
    'driver_actions/safe_drive',
    'driver_actions/change_gear',  # This is hard to distinguish - one hand
    'hand_on_gear/hand_on_gear',  # This is hard to distinguish - one hand
    'moving_hands/not_moving',
    'moving_hands/moving',
    'transition/taking control',
    'transition/giving control',
]
SOURCE_TYPES: Final[list[str]] = ['rgb', 'depth', 'ir']
DRIVER_SESSION_MAPPING: Final[dict[int, list[str]]] = {
    1: [
        'gA_1_s1_2019-03-08T09;31;15+01;00',
        'gA_1_s2_2019-03-08T09;21;03+01;00',
        'gA_1_s3_2019-03-14T14;31;08+01;00',
        'gA_1_s4_2019-03-22T11;49;58+01;00',
        'gA_1_s5_2019-03-14T14;26;17+01;00',
        'gA_1_s6_2019-03-08T09;15;15+01;00',
        'gA_1_s6_2019-03-22T11;59;56+01;00',
    ],
    2: [
        'gA_2_s1_2019-03-08T10;01;44+01;00',
        'gA_2_s2_2019-03-08T09;50;49+01;00',
        'gA_2_s3_2019-03-13T09;23;42+01;00',
        'gA_2_s4_2019-03-22T09;15;55+01;00',
        'gA_2_s5_2019-03-13T09;19;23+01;00',
        'gA_2_s6_2019-03-08T09;44;39+01;00',
        'gA_2_s6_2019-03-22T09;25;44+01;00',
    ],
    3: [
        'gA_3_s1_2019-03-08T10;27;38+01;00',
        'gA_3_s2_2019-03-08T10;16;48+01;00',
        'gA_3_s3_2019-03-13T09;41;01+01;00',
        'gA_3_s4_2019-03-22T10;15;27+01;00',
        'gA_3_s5_2019-03-13T09;36;25+01;00',
        'gA_3_s6_2019-03-08T10;12;10+01;00',
        'gA_3_s6_2019-03-22T10;25;17+01;00',
    ],
    4: [
        'gA_4_s1_2019-03-13T10;36;15+01;00',
        'gA_4_s2_2019-03-13T10;43;06+01;00',
        'gA_4_s3_2019-03-13T11;00;49+01;00',
        'gA_4_s4_2019-03-25T11;44;29+01;00',
        'gA_4_s5_2019-03-13T10;56;52+01;00',
        'gA_4_s6_2019-03-13T10;51;44+01;00',
        'gA_4_s6_2019-03-25T11;54;30+01;00',
    ],
    5: [
        'gA_5_s1_2019-03-08T10;57;00+01;00',
        'gA_5_s2_2019-03-08T10;46;46+01;00',
        'gA_5_s3_2019-03-13T09;10;35+01;00',
        'gA_5_s4_2019-03-22T11;28;00+01;00',
        'gA_5_s5_2019-03-13T09;06;49+01;00',
        'gA_5_s6_2019-03-08T10;40;35+01;00',
        'gA_5_s6_2019-03-22T11;37;16+01;00',
    ],
}


def get_all_sessions() -> list[str]:
    return sorted(
        [
            session
            for sessions in DRIVER_SESSION_MAPPING.values()
            for session in sessions
        ]
    )


def get_clips(session: Path, category: str) -> list[Path]:
    return [x for x in (session / category).glob('*') if x.is_dir()]


def get_frame_paths(
    session: Path, category: str, source_type: str, extension: str
) -> list[Path]:
    sequences = get_clips(session, category)
    return sorted(
        [img for x in sequences for img in (x / source_type).glob(f'*.{extension}')],
        key=lambda x: int(x.stem),
    )


@cache
def _load_annotations(annotations_file_path: str | Path) -> dict:
    with open(annotations_file_path) as f:
        return json.load(f)


def _setup_output_directories(
    base_dir: str | Path,
    force_overwrite: bool = False,
    skip_output_directory_setup: bool = False,
) -> tuple[Path, Path]:
    """Setup output directories for normal and anomalous/distraction frames."""
    base_dir = Path(base_dir)
    normal_dir = base_dir / 'normal'
    anomaly_dir = base_dir / 'anomal'

    if skip_output_directory_setup:
        return normal_dir, anomaly_dir

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
    source_type: Literal['rgb', 'depth', 'ir'],
    output_base_dir: str | Path | None = None,
    target_size: tuple[int, int] | None = None,
    force_overwrite: bool = False,
    depth_threshold: int = 2000,
    skip_output_directory_setup: bool = False,
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
    depth_threshold : int
        Depth threshold for depth videos in millimeters. Values above this threshold will be clipped.

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
        f'Unknown actions found in annotations file: {set(actions.values()) - set(known_actions)}'
    )

    distraction_mapping = _get_distraction_mapping(annotations_file_path)

    # Setup output directories
    normal_dir, anomaly_dir = _setup_output_directories(
        output_base_dir,
        force_overwrite=force_overwrite,
        skip_output_directory_setup=skip_output_directory_setup,
    )

    # Initialize video capture
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f'Failed to open video file: {input_video_path}')

    if source_type == 'depth':
        # Disable RGB conversion for depth videos
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        export_extension = 'png'
    elif source_type == 'ir':
        export_extension = 'png'
    else:
        export_extension = 'jpg'

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

            if source_type == 'depth':
                frame = preprocess_depth_frame(frame, depth_threshold=depth_threshold)
                target_dir = 'source_depth'
            else:
                target_dir = source_type
                if source_type == 'ir':
                    # Convert IR frames to grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if source_type == 'depth' or source_type == 'ir':
                frame = crop_aspect_ratio(frame, 4 / 3)

            # Resize frame if target size is specified
            if target_size is not None:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

            frame_key = str(frame_index)
            is_anomaly = False
            frame_annotations = frames_data.get(frame_key)
            if frame_annotations:
                frame_actions = frame_annotations.get('actions')
                if frame_actions:
                    for action_key, _ in frame_actions.items():
                        if distraction_mapping[int(action_key)] == 1:
                            # One of the annotated actions for the frame is a distraction
                            is_anomaly = True
                            break

            # Create new sequence directory if anomaly status changed
            if previous_is_anomaly is None or is_anomaly != previous_is_anomaly:
                current_sequence += 1
                base_dir = anomaly_dir if is_anomaly else normal_dir
                current_sequence_dir = (
                    base_dir / f'sequence_{current_sequence}' / target_dir
                )
                current_sequence_dir.mkdir(exist_ok=True, parents=True)
                previous_is_anomaly = is_anomaly

            # Save frame
            if current_sequence_dir is not None:
                pbar.set_postfix_str(f'Sequence: {current_sequence}')
                frame_path = (
                    current_sequence_dir / f'{frame_index:06d}.{export_extension}'
                )
                cv2.imwrite(str(frame_path), frame)

            frame_index += 1
            pbar.update(1)

    # Release resources
    cap.release()


def _load_images_as_array(
    input_dir: str | Path, extension: str = 'jpg', limit: None | int = None
) -> np.ndarray:
    """Load images from directory as a NumPy array."""
    input_dir = Path(input_dir)
    image_files = sorted(input_dir.glob(f'*.{extension}'), key=lambda p: p.stem)
    if limit is not None:
        image_files = image_files[:limit]
    images = [np.array(Image.open(img)) for img in image_files]
    return np.stack(images, axis=0)


def convert_depth_images_to_video(
    session: str,
    category: Literal['normal', 'anomal'],
    sequence: int,
    directory: str = 'depth',
    fps: int = 10,
    extension: str = 'jpg',
) -> None:
    from .depth_anything.utils.dc_utils import save_video

    path = ROOT / session / category / f'sequence_{sequence}' / directory
    if not path.exists():
        raise ValueError(f'Directory not found: {path}')
    frames = _load_images_as_array(path, extension=extension)
    save_video(frames, path / 'video_depth.mp4', fps=fps, is_depths=True)


def convert_frames_to_video(
    session: str,
    source_type: Literal['rgb', 'depth'],
    preset: Literal[
        'ultrafast',
        'superfast',
        'veryfast',
        'faster',
        'fast',
        'medium',
        'slow',
        'slower',
        'veryslow',
    ] = 'medium',
    crf: int = 0,
    crop_driver: bool = True,
    resize: tuple[int, int] = (518, 518),
    extension: str = 'jpg',
) -> None:
    """Convert extracted frames to video. Assumes that function `extract_frames` has been called."""
    from sam2util import convert_images_to_mp4

    base_dir = ROOT / session
    if not base_dir.exists():
        raise ValueError(f'Session directory not found: {base_dir}')

    all_sequences = sorted([p for p in base_dir.rglob(source_type) if p.is_dir()])
    pprint(all_sequences)

    output_help_dir = f'crop_{source_type}'

    for seq_dir in (pbar := tqdm(all_sequences)):
        pbar.set_postfix_str(seq_dir.parent.name)

        source_dir = seq_dir
        if crop_driver:
            source_dir = seq_dir.parent / output_help_dir
            source_dir.mkdir(exist_ok=True, parents=True)
            for img_file in seq_dir.glob(f'*.{extension}'):
                img = Image.open(img_file)
                img = pad_to_square(img).resize(resize)
                img.save(source_dir / img_file.name)

        output_file = source_dir / 'video.mp4'
        convert_images_to_mp4(
            image_folder=source_dir,
            output_video_path=output_file,
            preset=preset,
            crf=crf,
        )
