from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

FONT_DIR = Path(__file__).parent.parent / 'resources' / 'fonts'


def _set_font(font_name: str, type: str) -> FontProperties:
    fpath = FONT_DIR / font_name
    fontManager.addfont(fpath)
    font = FontProperties(fname=fpath)
    plt.rcParams['font.' + type].append(font.get_name())
    plt.rcParams['font.family'] = font.get_name()
    return font


def set_cmu_serif_font(bold: bool = False) -> FontProperties:
    return (
        _set_font('cmunbx.ttf', 'serif') if bold else _set_font('cmunrm.ttf', 'serif')
    )


def set_cmu_typewriter_font() -> FontProperties:
    return _set_font('cmuntt.ttf', 'monospace')


def set_cmu_sans_serif_font() -> FontProperties:
    return _set_font('cmunss.ttf', 'sans-serif')
