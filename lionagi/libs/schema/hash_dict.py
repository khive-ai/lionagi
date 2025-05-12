import orjson as json
from pydantic import BaseModel


def hash_dict(data) -> int:
    hashable_items = []
    if isinstance(data, BaseModel):
        data = data.model_dump()
    for k, v in data.items():
        if isinstance(v, (list, dict)):
            # Convert unhashable types to JSON string for hashing
            v = json.dumps(v, sort_keys=True)
        elif not isinstance(v, (str, int, float, bool, type(None))):
            # Convert other unhashable types to string representation
            v = str(v)
        hashable_items.append((k, v))
    return hash(frozenset(hashable_items))
