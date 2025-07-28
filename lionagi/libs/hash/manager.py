class HashUtils:
    @staticmethod
    def hash_dict(data: any, strict: bool = False) -> int:
        """
        Computes a deterministic hash for various Python data structures including
        dictionaries, Pydantic BaseModels, lists, tuples, sets, frozensets, and primitives.

        The hash is deterministic within the same Python process run (respecting
        PYTHONHASHSEED for built-in hash behavior on strings, bytes, etc.).
        It's suitable for tasks like finding unique objects within a collection
        during a single program execution.

        Args:
            data: The Python object to hash.
            strict: if True, will make a deep copy of the input data to ensure immutability.

        Returns:
            An integer hash value.

        Raises:
            TypeError: If the generated internal representation of the data is not hashable,
                    though this is unlikely with the current _generate_hashable_representation.
        """
        from .hash_dict import hash_dict as _hash_dict

        return _hash_dict(data, strict=strict)
