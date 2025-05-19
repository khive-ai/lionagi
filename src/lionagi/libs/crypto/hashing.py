# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import orjson 
import json 
from pydantic import BaseModel 
from typing import Any

__all__ = ("sha256_of_dict", "hash_dict")

def sha256_of_dict(obj: dict) -> str:
    """Deterministic SHA-256 of an arbitrary mapping."""
    
    payload: bytes = orjson.dumps(
        obj,
        option=(
            orjson.OPT_SORT_KEYS
            | orjson.OPT_NON_STR_KEYS
        ),
    )
    return hashlib.sha256(memoryview(payload)).hexdigest()

def hash_dict(data: Any) -> int:
    """
    Generates a hash for a dictionary, serializing unhashable items.
    Handles Pydantic models by converting them to dicts first.
    Sorts dictionary items to ensure deterministic hash.
    """
    hashable_items = []
    
    if isinstance(data, BaseModel):
        data = data.model_dump()
    
    if not isinstance(data, dict):
        try:
            return hash(data)
        except TypeError:
            return hash(str(data))

    # Sort items by key for deterministic hashing
    for k, v in sorted(data.items()): 
        if isinstance(v, list): # Convert lists to tuple of potentially stringified items
            v_tuple = tuple(json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else item for item in v)
            v = v_tuple
        elif isinstance(v, dict): # For nested dicts, recursively call or stringify
             v = json.dumps(v, sort_keys=True) # Simple approach: stringify nested dicts
        elif not isinstance(v, (str, int, float, bool, type(None))):
            v = str(v) 
        hashable_items.append((k, v))
    return hash(frozenset(hashable_items))