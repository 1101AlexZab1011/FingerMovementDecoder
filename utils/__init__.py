import mne
from typing import Optional


def info_pick_channels(info: mne.Info, ch_names: list[str], ordered: Optional[bool] = False) -> mne.Info:
    sel = mne.pick_channels(info.ch_names, ch_names)
    return mne.pick_info(info, sel, copy=False, verbose=False)