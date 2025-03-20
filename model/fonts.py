from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

FONT_DIR = Path(__file__).parent.parent / 'resources' / 'fonts'


def _set_font(font_name: str) -> FontProperties:
    fpath = FONT_DIR / font_name
    fontManager.addfont(fpath)
    font = FontProperties(fname=fpath)
    plt.rcParams['font.family'] = font.get_name()
    return font


def set_cmu_serif_font() -> FontProperties:
    return _set_font('cmunrm.ttf')


def set_cmu_typewriter_font() -> FontProperties:
    return _set_font('cmuntt.ttf')
