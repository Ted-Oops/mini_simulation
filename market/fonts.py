import os
from functools import lru_cache
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font

FONT_SAMPLE = "市场模拟实验报告基本面动量投机噪声策略规则半开放式决策价格路径净需求成交量买入卖出观望约束阻止最终财富均值智能体轮次预热网格"


@lru_cache(maxsize=4)
def resolve_chinese_font(override: str | None = None) -> str:
    candidates = []
    if override:
        candidates.append(override)
    else:
        windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        candidates.extend(str(windows / name) for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc"))
        names = ("Microsoft YaHei", "SimHei", "Noto Sans CJK", "Source Han Sans", "WenQuanYi", "PingFang", "Heiti")
        candidates.extend(font.fname for font in font_manager.fontManager.ttflist
                          if any(name in font.name for name in names))
    required = {ord(char) for char in FONT_SAMPLE}
    for filename in dict.fromkeys(candidates):
        if not Path(filename).is_file():
            continue
        try:
            if required.issubset(FT2Font(filename).get_charmap()):
                return filename
        except (OSError, RuntimeError):
            continue
    raise RuntimeError("未找到能显示中文的字体，请安装微软雅黑或思源黑体，或用 MINI_SIM_CJK_FONT 指定字体文件路径。")


def configure_chinese_font() -> str:
    filename = resolve_chinese_font(os.getenv("MINI_SIM_CJK_FONT"))
    font_manager.fontManager.addfont(filename)
    name = font_manager.FontProperties(fname=filename).get_name()
    matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return filename
