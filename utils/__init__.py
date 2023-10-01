import mne
from typing import Optional


def info_pick_channels(info: mne.Info, ch_names: list[str], ordered: Optional[bool] = None) -> mne.Info:
    """
    Pick specific channels from an MNE Info object and return a new Info object.

    This function selects a subset of channels from the input MNE Info object based on a list of channel names.

    Args:
        info (mne.Info): The MNE Info object containing channel information.
        ch_names (list[str]): A list of channel names to select from the Info object.
        ordered (bool, optional): If True, preserve the order of channels as specified in 'ch_names'. 
            If False (default), the output Info object will have channels in the order of the original Info object.

    Returns:
        mne.Info: A new MNE Info object containing only the selected channels.

    Raises:
        ValueError: If any channel name in 'ch_names' is not found in the original Info object.

    Notes:
        - The input 'info' object remains unchanged, and the function returns a new Info object with the selected channels.
        - If 'ordered' is True, the order of channels in the returned Info object will match the order in 'ch_names'.
        - If 'ordered' is False (default), the order will match the original Info object.

    Examples:
        To select specific channels from an Info object 'info' and preserve their order:

        >>> selected_channels = ['EEG 001', 'EEG 002', 'EEG 003']
        >>> new_info = info_pick_channels(info, selected_channels, ordered=True)

        To select specific channels and retain the original order:

        >>> selected_channels = ['EEG 002', 'EEG 003', 'EEG 001']
        >>> new_info = info_pick_channels(info, selected_channels)
    """
    sel = mne.pick_channels(info.ch_names, ch_names, ordered=ordered)
    return mne.pick_info(info, sel, copy=False, verbose=False)