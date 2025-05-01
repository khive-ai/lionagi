import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from lionagi.scripts.build_adapter_registry import (
    extract_obj_key,
    find_adapter_classes,
    is_adapter_class,
    scan_adapters_directory,
)


def test_is_adapter_class():
    """Test the is_adapter_class function with a class that has an obj_key attribute."""
    # Create a simple AST node for a class with an obj_key attribute
    import ast

    class_def = ast.parse(
        """
class TestAdapter:
    obj_key = "test"
    """
    ).body[0]

    assert is_adapter_class(class_def)


def test_is_adapter_class_negative():
    """Test the is_adapter_class function with a class that doesn't have an obj_key attribute."""
    # Create a simple AST node for a class without an obj_key attribute
    import ast

    class_def = ast.parse(
        """
class NotAnAdapter:
    some_attr = "test"
    """
    ).body[0]

    assert not is_adapter_class(class_def)


def test_extract_obj_key():
    """Test the extract_obj_key function."""
    # Create a simple AST node for a class with an obj_key attribute
    import ast

    class_def = ast.parse(
        """
class TestAdapter:
    obj_key = "test"
    """
    ).body[0]

    assert extract_obj_key(class_def) == "test"


def test_extract_obj_key_negative():
    """Test the extract_obj_key function with a class that doesn't have an obj_key attribute."""
    # Create a simple AST node for a class without an obj_key attribute
    import ast

    class_def = ast.parse(
        """
class NotAnAdapter:
    some_attr = "test"
    """
    ).body[0]

    assert extract_obj_key(class_def) is None


def test_find_adapter_classes():
    """Test the find_adapter_classes function."""
    # Create a temporary file with a class that has an obj_key attribute
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".py", delete=False
    ) as temp_file:
        temp_file.write(
            """
class TestAdapter:
    obj_key = "test"

class AnotherAdapter:
    obj_key = "another"

class NotAnAdapter:
    some_attr = "test"
"""
        )
        temp_file_path = temp_file.name

    try:
        # Find adapter classes in the temporary file
        adapter_classes = find_adapter_classes(temp_file_path)

        # Verify the adapter classes were found
        assert len(adapter_classes) == 2
        assert ("TestAdapter", "test") in adapter_classes
        assert ("AnotherAdapter", "another") in adapter_classes
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def test_scan_adapters_directory():
    """Test the scan_adapters_directory function."""
    # Create a temporary directory with adapter files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a subdirectory
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)

        # Create adapter files
        with open(os.path.join(temp_dir, "adapter1.py"), "w") as f:
            f.write(
                """
class Adapter1:
    obj_key = "adapter1"
"""
            )

        with open(os.path.join(subdir, "adapter2.py"), "w") as f:
            f.write(
                """
class Adapter2:
    obj_key = "adapter2"
"""
            )

        # Create a non-adapter file
        with open(os.path.join(temp_dir, "not_adapter.py"), "w") as f:
            f.write(
                """
class NotAnAdapter:
    some_attr = "test"
"""
            )

        # Create a file that should be ignored
        with open(os.path.join(temp_dir, "__init__.py"), "w") as f:
            f.write("")

        # Scan the temporary directory
        adapter_map = scan_adapters_directory(temp_dir)

        # Verify the adapter map
        assert len(adapter_map) == 2
        assert "adapter1" in adapter_map
        assert "adapter2" in adapter_map
        assert adapter_map["adapter1"].endswith(".Adapter1")
        assert adapter_map["adapter2"].endswith("subdir.adapter2.Adapter2")


def test_main_function():
    """Test the main function of the build_adapter_registry script."""
    # Create a temporary directory with adapter files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create adapter files
        with open(os.path.join(temp_dir, "adapter1.py"), "w") as f:
            f.write(
                """
class Adapter1:
    obj_key = "adapter1"
"""
            )

        # Create a temporary output file
        output_file = os.path.join(temp_dir, "adapter_map.json")

        # Mock the command-line arguments
        with patch(
            "sys.argv",
            [
                "build_adapter_registry",
                "--adapters-dir",
                temp_dir,
                "--output",
                output_file,
            ],
        ):
            # Import and run the main function
            from lionagi.scripts.build_adapter_registry import main

            main()

        # Verify the output file was created
        assert os.path.exists(output_file)

        # Verify the content of the output file
        with open(output_file, "r") as f:
            adapter_map = json.load(f)
            assert len(adapter_map) == 1
            assert "adapter1" in adapter_map
            assert adapter_map["adapter1"].endswith(".Adapter1")
