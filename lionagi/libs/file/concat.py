import asyncio
from pathlib import Path
from typing import Any, Callable, Coroutine

import aiofiles

from lionagi.utils import create_path, lcall

from .file_ops import async_read_file
from .process import dir_to_files, async_dir_to_files


def concat(
    data_path: str | Path | list,
    file_types: list[str],
    output_dir: str | Path = None,
    output_filename: str = None,
    file_exist_ok: bool = True,
    recursive: bool = True,
    verbose: bool = True,
    threshold: int = 0,
    return_fps: bool = False,
    return_files: bool = False,
    exclude_patterns: list[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """
    data_path: str or Path or list of str or Path, the directory or file paths to concatenate.
    file_types: list of str, the file types to concatenate. [e.g. ['.txt', '.md']]
    output_dir: str or Path, the directory to save the concatenated file. If provided, will save the file.
    output_filename: str, the filename to save the concatenated file.
    file_exist_ok: bool, if True, overwrite the existing file. Default is True.
    recursive: bool, if True, search files recursively. Default is True.
    verbose: bool, if True, print the output path. Default is True.
    threshold: int, the minimum number of chars for the file to be considered valid to concatenate.
    exclude_patterns: list of str, patterns to exclude files from concatenation (e.g. ['log', 'temp', '.venv']).
    kwargs: additional keyword arguments to pass to create_path.
    """
    persist_path = None
    if output_dir:
        if not output_filename:
            output_filename = "concatenated_text.txt"
            kwargs["timestamp"] = kwargs.get("timestamp", True)
            kwargs["random_hash_digits"] = kwargs.get("random_hash_digits", 6)
        output_filename = output_filename or "concatenated_text.txt"
        persist_path = create_path(
            output_dir, output_filename, file_exist_ok=file_exist_ok, **kwargs
        )

    texts = []

    def _check_existence(_p: str) -> Path | list[Path] | None:
        if exclude_patterns:
            _str_p = str(_p)
            for pattern in exclude_patterns:
                if pattern in _str_p:
                    return None

        if not Path(_p).exists():
            # if the path doesn't exist, return None
            if verbose:
                print(f"Path {_p} does not exist, skipping...")
            return None

        p = Path(_p)
        if p.is_dir():
            return dir_to_files(
                p,
                recursive=recursive,
                file_types=file_types,
                ignore_errors=True,
                max_workers=5,
            )
        if p.is_file():
            return p

    data_path: list[Path] = lcall(
        data_path,
        _check_existence,
        sanitize_input=True,
        unique_input=True,
        flatten=True,
        dropna=True,
        unique_output=True,
        flatten_tuple_set=True,
    )

    contents = {}
    fps = []
    for dp in data_path:
        try:
            text = dp.read_text(encoding="utf-8")

        except Exception:
            # if we cannot read the file, skip it
            print(f"Could not read file: {dp}. Skipping...")
            continue

        if threshold > 0 and len(text) < threshold:
            continue

        fps.append(dp)
        contents[str(dp)] = text

    for k, text in sorted(contents.items(), key=lambda x: x[0]):
        texts.extend(["---", k, "---\n", text])

    text = "\n".join(texts)
    if persist_path:
        persist_path.write_text(text, encoding="utf-8")
    if verbose:
        print(
            f"Concatenated {len(fps)} files to {persist_path}."
            f" The file contains {len(text)} characters."
        )

    out = {"text": text}  # default output
    if persist_path:
        out["persist_path"] = persist_path
    if return_files:
        out["texts"] = texts
    if return_fps:
        out["fps"] = fps

    return out


async def async_concat(
    data_path: str | Path | list,
    file_types: list[str],
    output_dir: str | Path = None,
    output_filename: str = None,
    file_exist_ok: bool = True,
    recursive: bool = True,
    verbose: bool = True,
    threshold: int = 0,
    return_fps: bool = False,
    return_files: bool = False,
    exclude_patterns: list[str] = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Asynchronously concatenate files from specified paths.
    
    Args:
        data_path: str or Path or list of str or Path, the directory or file paths to concatenate.
        file_types: list of str, the file types to concatenate. [e.g. ['.txt', '.md']]
        output_dir: str or Path, the directory to save the concatenated file. If provided, will save the file.
        output_filename: str, the filename to save the concatenated file.
        file_exist_ok: bool, if True, overwrite the existing file. Default is True.
        recursive: bool, if True, search files recursively. Default is True.
        verbose: bool, if True, print the output path. Default is True.
        threshold: int, the minimum number of chars for the file to be considered valid to concatenate.
        exclude_patterns: list of str, patterns to exclude files from concatenation (e.g. ['log', 'temp', '.venv']).
        kwargs: additional keyword arguments to pass to create_path.
    
    Returns:
        dict: A dictionary containing the concatenated text and optionally file paths and other metadata.
    """
    persist_path = None
    if output_dir:
        if not output_filename:
            output_filename = "concatenated_text.txt"
            kwargs["timestamp"] = kwargs.get("timestamp", True)
            kwargs["random_hash_digits"] = kwargs.get("random_hash_digits", 6)
        output_filename = output_filename or "concatenated_text.txt"
        persist_path = create_path(
            output_dir, output_filename, file_exist_ok=file_exist_ok, **kwargs
        )

    texts = []

    async def _async_check_existence(_p: str) -> Path | list[Path] | None:
        if exclude_patterns:
            _str_p = str(_p)
            for pattern in exclude_patterns:
                if pattern in _str_p:
                    return None

        p = Path(_p)
        if not p.exists():
            # if the path doesn't exist, return None
            if verbose:
                print(f"Path {_p} does not exist, skipping...")
            return None

        if p.is_dir():
            return await async_dir_to_files(
                p,
                recursive=recursive,
                file_types=file_types,
                ignore_errors=True,
                max_workers=5,
            )
        if p.is_file():
            return p

    # Process paths in parallel
    if isinstance(data_path, list):
        path_tasks = [_async_check_existence(p) for p in data_path]
        path_results = await asyncio.gather(*path_tasks, return_exceptions=True)
        
        # Filter out exceptions and None values
        data_path = []
        for result in path_results:
            if isinstance(result, Exception):
                if verbose:
                    print(f"Error processing path: {result}")
            elif result is not None:
                if isinstance(result, list):
                    data_path.extend(result)
                else:
                    data_path.append(result)
    else:
        result = await _async_check_existence(data_path)
        if isinstance(result, list):
            data_path = result
        elif result is not None:
            data_path = [result]
        else:
            data_path = []
    
    # Make sure data_path is a list of Path objects
    if not isinstance(data_path, list):
        data_path = [data_path]
    
    # Ensure all items are Path objects
    data_path = [Path(p) if not isinstance(p, Path) else p for p in data_path]
    
    # Apply exclude patterns to the file paths
    if exclude_patterns:
        filtered_data_path = []
        for p in data_path:
            should_exclude = False
            str_p = str(p)
            for pattern in exclude_patterns:
                if pattern in str_p:
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_data_path.append(p)
        data_path = filtered_data_path
    
    # Remove duplicates and sort
    data_path = sorted(set(data_path), key=lambda x: str(x))

    contents = {}
    fps = []
    
    # Create tasks for reading all files concurrently
    read_tasks = []
    for dp in data_path:
        read_tasks.append((dp, asyncio.create_task(async_read_file(dp))))
    
    # Process the results as they complete
    for dp, task in read_tasks:
        try:
            text = await task
            if threshold > 0 and len(text) < threshold:
                continue
                
            fps.append(dp)
            contents[str(dp)] = text
        except Exception as e:
            # if we cannot read the file, skip it
            if verbose:
                print(f"Could not read file: {dp}. Error: {e}. Skipping...")

    for k, text in sorted(contents.items(), key=lambda x: x[0]):
        texts.extend(["---", k, "---\n", text])

    text = "\n".join(texts)
    if persist_path:
        async with aiofiles.open(persist_path, "w", encoding="utf-8") as f:
            await f.write(text)
    if verbose:
        print(
            f"Concatenated {len(fps)} files to {persist_path}."
            f" The file contains {len(text)} characters."
        )

    out = {"text": text}  # default output
    if persist_path:
        out["persist_path"] = persist_path
    if return_files:
        out["texts"] = texts
    if return_fps:
        out["fps"] = fps

    return out
