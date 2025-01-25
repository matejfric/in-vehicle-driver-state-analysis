from pathlib import Path
from typing import Final

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
