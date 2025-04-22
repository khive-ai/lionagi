import os
import pytest
import tempfile
from pathlib import Path

from lionagi.libs.file.concat_files import async_concat_files


@pytest.mark.asyncio
async def test_async_concat_files():
    """Test async_concat_files function."""
    # Create a temporary directory with some files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create files with different content
        file1_path = os.path.join(temp_dir, "file1.txt")
        file2_path = os.path.join(temp_dir, "file2.txt")
        
        with open(file1_path, "w") as f:
            f.write("Content of file 1")
        with open(file2_path, "w") as f:
            f.write("Content of file 2")
        
        # Test concatenating files without saving
        result = await async_concat_files(
            data_path=[file1_path, file2_path],
            file_types=[".txt"],
            verbose=False,
        )
        
        assert "Content of file 1" in result
        assert "Content of file 2" in result
        
        # Test concatenating files with saving
        with tempfile.TemporaryDirectory() as output_dir:
            result, fps = await async_concat_files(
                data_path=[file1_path, file2_path],
                file_types=[".txt"],
                output_dir=output_dir,
                output_filename="concatenated.txt",
                verbose=False,
                return_fps=True,
            )
            
            # Verify the output file exists
            output_file = os.path.join(output_dir, "concatenated.txt")
            assert os.path.exists(output_file)
            
            # Verify the content of the output file
            with open(output_file, "r") as f:
                content = f.read()
                assert "Content of file 1" in content
                assert "Content of file 2" in content
            
            # Verify the returned file paths
            assert len(fps) == 2
            assert Path(file1_path) in fps
            assert Path(file2_path) in fps


@pytest.mark.asyncio
async def test_async_concat_files_with_directory():
    """Test async_concat_files function with a directory as input."""
    # Create a temporary directory with some files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create files with different content and extensions
        file1_path = os.path.join(temp_dir, "file1.txt")
        file2_path = os.path.join(temp_dir, "file2.txt")
        file3_path = os.path.join(temp_dir, "file3.md")
        
        with open(file1_path, "w") as f:
            f.write("Content of file 1")
        with open(file2_path, "w") as f:
            f.write("Content of file 2")
        with open(file3_path, "w") as f:
            f.write("Content of file 3")
        
        # Test concatenating files from a directory
        result = await async_concat_files(
            data_path=temp_dir,
            file_types=[".txt"],
            verbose=False,
        )
        
        assert "Content of file 1" in result
        assert "Content of file 2" in result
        assert "Content of file 3" not in result  # Should not include .md files


@pytest.mark.asyncio
async def test_async_concat_files_with_threshold():
    """Test async_concat_files function with a threshold."""
    # Create a temporary directory with some files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create files with different content lengths
        file1_path = os.path.join(temp_dir, "file1.txt")
        file2_path = os.path.join(temp_dir, "file2.txt")
        
        with open(file1_path, "w") as f:
            f.write("Short content")  # 13 characters
        with open(file2_path, "w") as f:
            f.write("This is a longer content that should be included")  # 45 characters
        
        # Test concatenating files with a threshold
        result = await async_concat_files(
            data_path=[file1_path, file2_path],
            file_types=[".txt"],
            threshold=20,  # Only include files with more than 20 characters
            verbose=False,
        )
        
        assert "Short content" not in result
        assert "This is a longer content that should be included" in result


@pytest.mark.asyncio
async def test_async_concat_files_return_files():
    """Test async_concat_files function with return_files=True."""
    # Create a temporary directory with some files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create files with different content
        file1_path = os.path.join(temp_dir, "file1.txt")
        file2_path = os.path.join(temp_dir, "file2.txt")
        
        with open(file1_path, "w") as f:
            f.write("Content of file 1")
        with open(file2_path, "w") as f:
            f.write("Content of file 2")
        
        # Test concatenating files with return_files=True
        result = await async_concat_files(
            data_path=[file1_path, file2_path],
            file_types=[".txt"],
            verbose=False,
            return_files=True,
        )
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert any("Content of file 1" in text for text in result)
        assert any("Content of file 2" in text for text in result)


@pytest.mark.performance
async def test_sync_vs_async_concat_files_performance():
    """Test performance comparison between sync and async concat_files."""
    import asyncio
    import time
    from lionagi.libs.file.concat_files import concat_files

    # Create a temporary directory with multiple files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create 50 files with content
        file_paths = []
        for i in range(50):
            file_path = os.path.join(temp_dir, f"file{i}.txt")
            with open(file_path, "w") as f:
                f.write(f"Content of file {i}" * 100)  # Make it reasonably large
            file_paths.append(file_path)
        
        # Test synchronous concat_files
        start_time = time.time()
        concat_files(
            data_path=file_paths,
            file_types=[".txt"],
            verbose=False,
        )
        sync_time = time.time() - start_time
        
        # Test asynchronous concat_files
        start_time = time.time()
        await async_concat_files(
            data_path=file_paths,
            file_types=[".txt"],
            verbose=False,
        )
        async_time = time.time() - start_time
        
        # For small files or in certain environments, async might have more overhead
        # We're just printing the times for information, not asserting
        print(f"Sync time: {sync_time:.4f}s, Async time: {async_time:.4f}s")
        
        # The real benefit of async I/O is seen with many concurrent operations
        # or when I/O operations are slow (like network requests)
        # For local file operations, the difference might not be as pronounced


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])