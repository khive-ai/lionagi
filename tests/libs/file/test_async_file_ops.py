import os
import pytest
import tempfile
from pathlib import Path

from lionagi.libs.file.file_ops import (
    async_copy_file,
    async_get_file_size,
    async_list_files,
    async_read_file,
)


@pytest.mark.asyncio
async def test_async_read_file():
    """Test async_read_file function."""
    # Create a temporary file with some content
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_file.write("Test content")
        temp_file_path = temp_file.name

    try:
        # Test reading the file
        content = await async_read_file(temp_file_path)
        assert content == "Test content"

        # Test file not found error
        with pytest.raises(FileNotFoundError):
            await async_read_file("nonexistent_file.txt")
    finally:
        # Clean up
        os.unlink(temp_file_path)


@pytest.mark.asyncio
async def test_async_copy_file():
    """Test async_copy_file function."""
    # Create a temporary file with some content
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_file.write("Test content")
        temp_file_path = temp_file.name

    # Create a temporary directory for the destination
    with tempfile.TemporaryDirectory() as temp_dir:
        dest_path = os.path.join(temp_dir, "copied_file.txt")

        try:
            # Test copying the file
            await async_copy_file(temp_file_path, dest_path)
            assert os.path.exists(dest_path)
            with open(dest_path, "r") as f:
                assert f.read() == "Test content"

            # Test file not found error
            with pytest.raises(FileNotFoundError):
                await async_copy_file("nonexistent_file.txt", dest_path)
        finally:
            # Clean up
            os.unlink(temp_file_path)


@pytest.mark.asyncio
async def test_async_get_file_size():
    """Test async_get_file_size function."""
    # Create a temporary file with some content
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_file.write("Test content")
        temp_file_path = temp_file.name

    try:
        # Test getting the file size
        size = await async_get_file_size(temp_file_path)
        assert size == len("Test content")

        # Test file not found error
        with pytest.raises(FileNotFoundError):
            await async_get_file_size("nonexistent_file.txt")
    finally:
        # Clean up
        os.unlink(temp_file_path)


@pytest.mark.asyncio
async def test_async_list_files():
    """Test async_list_files function."""
    # Create a temporary directory with some files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create files with different extensions
        with open(os.path.join(temp_dir, "file1.txt"), "w") as f:
            f.write("Test content 1")
        with open(os.path.join(temp_dir, "file2.txt"), "w") as f:
            f.write("Test content 2")
        with open(os.path.join(temp_dir, "file3.md"), "w") as f:
            f.write("Test content 3")

        # Test listing all files
        files = await async_list_files(temp_dir)
        assert len(files) == 3

        # Test listing files with a specific extension
        txt_files = await async_list_files(temp_dir, extension="txt")
        assert len(txt_files) == 2
        assert all(f.suffix == ".txt" for f in txt_files)

        # Test directory not found error
        with pytest.raises(NotADirectoryError):
            await async_list_files(os.path.join(temp_dir, "file1.txt"))


@pytest.mark.performance
async def test_sync_vs_async_read_performance():
    """Test performance comparison between sync and async file reading."""
    import asyncio
    import time
    from lionagi.libs.file.file_ops import read_file

    # Create multiple temporary files - use more files for a more realistic test
    num_files = 50
    temp_files = []
    
    for i in range(num_files):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            temp_file.write(f"Test content {i}" * 1000)  # Make it reasonably large
            temp_files.append(temp_file.name)
    
    try:
        # Test synchronous reading
        start_time = time.time()
        for file_path in temp_files:
            read_file(file_path)
        sync_time = time.time() - start_time
        
        # Test asynchronous reading
        start_time = time.time()
        tasks = [async_read_file(file_path) for file_path in temp_files]
        await asyncio.gather(*tasks)
        async_time = time.time() - start_time
        
        # For small files or in certain environments, async might have more overhead
        # We're just printing the times for information, not asserting
        print(f"Sync time: {sync_time:.4f}s, Async time: {async_time:.4f}s")
        
        # The real benefit of async I/O is seen with many concurrent operations
        # or when I/O operations are slow (like network requests)
        # For local file operations, the difference might not be as pronounced
    finally:
        # Clean up
        for file_path in temp_files:
            os.unlink(file_path)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])