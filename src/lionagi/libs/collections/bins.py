# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, List

__all__ = ("HasLen", "Bin", "get_bins")

class HasLen(Protocol):
    def __len__(self) -> int: ...

Bin = List[int]

def get_bins(input_list: List[HasLen], /, upper_limit: int) -> List[Bin]:
    """Organizes indices of items into bins based on a cumulative upper limit length.

    Args:
        input_list: The list of items (each must support len()) to be binned.
        upper_limit: The cumulative length upper limit for each bin. Items larger
                     than this limit will be placed in their own bin.

    Returns:
        A list of bins, where each bin is a list of indices from the input list.
    """
    if not input_list:
        return []

    all_bins: List[Bin] = []
    current_bin_indices: Bin = []
    current_bin_len_sum = 0

    for idx, item in enumerate(input_list):
        item_len = len(item)

        if item_len > upper_limit:
            if current_bin_indices: # Finalize previous bin
                all_bins.append(current_bin_indices)
            all_bins.append([idx]) # Large item gets its own bin
            current_bin_indices = []
            current_bin_len_sum = 0
            continue

        if current_bin_len_sum + item_len <= upper_limit:
            current_bin_indices.append(idx)
            current_bin_len_sum += item_len
        else:
            if current_bin_indices: # Finalize current bin before starting new
                all_bins.append(current_bin_indices)
            current_bin_indices = [idx] # Start new bin with current item
            current_bin_len_sum = item_len

    if current_bin_indices: # Add the last processed bin
        all_bins.append(current_bin_indices)

    return all_bins