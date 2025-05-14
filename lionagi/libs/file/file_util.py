# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal


class FileUtil:

    @staticmethod
    def chunk_by_chars(
        text: str,
        chunk_size: int = 2048,
        overlap: float = 0,
        threshold: int = 256,
    ) -> list[str]:
        from .chunk import chunk_by_chars

        return chunk_by_chars(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            threshold=threshold,
        )

    @staticmethod
    def chunk_by_tokens(
        text: str,
        tokenizer: Callable[[str], list[str]] = str.split,
        chunk_size: int = 1024,
        overlap: float = 0,
        threshold: int = 256,
    ) -> list[str]:
        from .chunk import chunk_by_tokens

        return chunk_by_tokens(
            text,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
            threshold=threshold,
        )

    @staticmethod
    def chunk_content(
        content: str,
        chunk_by: Literal["chars", "tokens"] = "chars",
        tokenizer: Callable[[str], list[str]] = str.split,
        chunk_size: int = 1024,
        overlap: float = 0,
        threshold: int = 256,
    ) -> list[str]:
        from .chunk import chunk_content

        return chunk_content(
            content,
            chunk_by=chunk_by,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
            threshold=threshold,
        )

    @staticmethod
    def concat_files(
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
        **kwargs,
    ) -> (
        list[str] | str | tuple[list[str], list[Path]] | tuple[str, list[Path]]
    ):
        from .concat_files import concat_files

        return concat_files(
            data_path,
            file_types=file_types,
            output_dir=output_dir,
            output_filename=output_filename,
            file_exist_ok=file_exist_ok,
            recursive=recursive,
            verbose=verbose,
            threshold=threshold,
            return_fps=return_fps,
            return_files=return_files,
            **kwargs,
        )

    @staticmethod
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
        from .concat import concat

        return concat(
            data_path,
            file_types=file_types,
            output_dir=output_dir,
            output_filename=output_filename,
            file_exist_ok=file_exist_ok,
            recursive=recursive,
            verbose=verbose,
            threshold=threshold,
            return_fps=return_fps,
            return_files=return_files,
            exclude_patterns=exclude_patterns,
            **kwargs,
        )

    @staticmethod
    def copy_file(src: Path | str, dest: Path | str) -> None:
        from .file_ops import copy_file

        copy_file(src, dest)

    @staticmethod
    def get_file_size(path: Path | str) -> int:
        from .file_ops import get_file_size

        return get_file_size(path)

    @staticmethod
    def list_files(
        dir_path: Path | str, extension: str | None = None
    ) -> list[Path]:
        from .file_ops import list_files

        return list_files(dir_path, extension=extension)

    @staticmethod
    def read_file(file_path: Path | str, encoding: str = "utf-8") -> str:
        from .file_ops import read_file

        return read_file(file_path, encoding=encoding)

    @staticmethod
    def read_image_to_base64(image_path: str | Path) -> str:
        import base64

        import cv2  # type: ignore[import]

        image_path = str(image_path)
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")

        file_extension = "." + image_path.split(".")[-1]

        success, buffer = cv2.imencode(file_extension, image)
        if not success:
            raise ValueError(
                f"Could not encode image to {file_extension} format."
            )
        encoded_image = base64.b64encode(buffer).decode("utf-8")
        return encoded_image

    @staticmethod
    def pdf_to_images(
        pdf_path: str, output_folder: str, dpi: int = 300, fmt: str = "jpeg"
    ) -> list:
        """
        Convert a PDF file into images, one image per page.

        Args:
            pdf_path (str): Path to the input PDF file.
            output_folder (str): Directory to save the output images.
            dpi (int): Dots per inch (resolution) for conversion (default: 300).
            fmt (str): Image format (default: 'jpeg'). Use 'png' if preferred.

        Returns:
            list: A list of file paths for the saved images.
        """
        import os

        from lionagi.utils import check_import

        convert_from_path = check_import(
            "pdf2image", import_name="convert_from_path"
        )

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Convert PDF to a list of PIL Image objects
        images = convert_from_path(pdf_path, dpi=dpi)

        saved_paths = []
        for i, image in enumerate(images):
            # Construct the output file name
            image_file = os.path.join(output_folder, f"page_{i + 1}.{fmt}")
            image.save(image_file, fmt.upper())
            saved_paths.append(image_file)

        return saved_paths

    @staticmethod
    def dir_to_files(
        directory: str | Path,
        file_types: list[str] | None = None,
        max_workers: int | None = None,
        ignore_errors: bool = False,
        verbose: bool = False,
        recursive: bool = False,
    ) -> list[Path]:
        from .process import dir_to_files

        return dir_to_files(
            directory,
            file_types=file_types,
            max_workers=max_workers,
            ignore_errors=ignore_errors,
            verbose=verbose,
            recursive=recursive,
        )

    @staticmethod
    def file_to_chunks(
        file_path: str | Path,
        chunk_by: Literal["chars", "tokens"] = "chars",
        chunk_size: int = 1500,
        overlap: float = 0.1,
        threshold: int = 200,
        encoding: str = "utf-8",
        custom_metadata: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        verbose: bool = False,
        timestamp: bool = True,
        random_hash_digits: int = 4,
        as_node: bool = False,
    ) -> list[dict[str, Any]]:
        from .file_ops import file_to_chunks

        return file_to_chunks(
            file_path,
            chunk_by=chunk_by,
            chunk_size=chunk_size,
            overlap=overlap,
            threshold=threshold,
            encoding=encoding,
            custom_metadata=custom_metadata,
            output_dir=output_dir,
            verbose=verbose,
            timestamp=timestamp,
            random_hash_digits=random_hash_digits,
            as_node=as_node,
        )

    @staticmethod
    def chunk(
        *,
        text: str | None = None,
        url_or_path: str | Path = None,
        file_types: list[str] | None = None,  # only local files
        recursive: bool = False,  # only local files
        tokenizer: Callable[[str], list[str]] = None,
        chunk_by: Literal["chars", "tokens"] = "chars",
        chunk_size: int = 1500,
        overlap: float = 0.1,
        threshold: int = 200,
        output_file: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        reader_tool: Callable = None,
        as_node: bool = False,
    ) -> list:
        from .process import chunk

        return chunk(
            text=text,
            url_or_path=url_or_path,
            file_types=file_types,
            recursive=recursive,
            tokenizer=tokenizer,
            chunk_by=chunk_by,
            chunk_size=chunk_size,
            overlap=overlap,
            threshold=threshold,
            output_file=output_file,
            metadata=metadata,
            reader_tool=reader_tool,
            as_node=as_node,
        )

    @staticmethod
    def save_to_file(
        text: str,
        directory: str | Path,
        filename: str,
        extension: str = "txt",
        timestamp: bool = True,
        dir_exist_ok: bool = True,
        file_exist_ok: bool = True,
        time_prefix: bool = False,
        timestamp_format: str = "%Y%m%d_%H%M%S",
        random_hash_digits: int = 4,
        verbose: bool = False,
    ) -> Path:
        from .save import save_to_file

        return save_to_file(
            text,
            directory=directory,
            filename=filename,
            extension=extension,
            timestamp=timestamp,
            dir_exist_ok=dir_exist_ok,
            file_exist_ok=file_exist_ok,
            time_prefix=time_prefix,
            timestamp_format=timestamp_format,
            random_hash_digits=random_hash_digits,
            verbose=verbose,
        )

    @staticmethod
    def create_path(
        directory: Path | str,
        filename: str,
        extension: str = None,
        timestamp: bool = False,
        dir_exist_ok: bool = True,
        file_exist_ok: bool = False,
        time_prefix: bool = False,
        timestamp_format: str | None = None,
        random_hash_digits: int = 0,
    ) -> Path:
        from .create_path import create_path

        return create_path(
            directory=directory,
            filename=filename,
            extension=extension,
            timestamp=timestamp,
            dir_exist_ok=dir_exist_ok,
            file_exist_ok=file_exist_ok,
            time_prefix=time_prefix,
            timestamp_format=timestamp_format,
            random_hash_digits=random_hash_digits,
        )
