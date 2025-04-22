import os
import pytest
import tempfile
from pathlib import Path

from lionagi.libs.file.concat import async_concat


@pytest.mark.asyncio
async def test_async_concat():
    """Test async_concat function."""
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
        result = await async_concat(
            data_path=[file1_path, file2_path],
            file_types=[".txt"],
            verbose=False,
        )
        
        assert "Content of file 1" in result["text"]
        assert "Content of file 2" in result["text"]
        
        # Test concatenating files with saving
        with tempfile.TemporaryDirectory() as output_dir:
            result = await async_concat(
                data_path=[file1_path, file2_path],
                file_types=[".txt"],
                output_dir=output_dir,
                output_filename="concatenated.txt",
                verbose=False,
                return_fps=True,
            )
            
            # Verify the output file exists
            assert "persist_path" in result
            assert result["persist_path"].exists()
            
            # Verify the content of the output file
            with open(result["persist_path"], "r") as f:
                content = f.read()
                assert "Content of file 1" in content
                assert "Content of file 2" in content
            
            # Verify the returned file paths
            assert "fps" in result
            assert len(result["fps"]) == 2
            assert Path(file1_path) in result["fps"]
            assert Path(file2_path) in result["fps"]


@pytest.mark.asyncio
async def test_async_concat_with_directory():
    """Test async_concat function with a directory as input."""
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
        result = await async_concat(
            data_path=temp_dir,
            file_types=[".txt"],
            verbose=False,
        )
        
        assert "Content of file 1" in result["text"]
        assert "Content of file 2" in result["text"]
        assert "Content of file 3" not in result["text"]  # Should not include .md files


@pytest.mark.asyncio
async def test_async_concat_with_threshold():
    """Test async_concat function with a threshold."""
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
        result = await async_concat(
            data_path=[file1_path, file2_path],
            file_types=[".txt"],
            threshold=20,  # Only include files with more than 20 characters
            verbose=False,
        )
        
        assert "Short content" not in result["text"]
        assert "This is a longer content that should be included" in result["text"]


@pytest.mark.asyncio
async def test_async_concat_with_exclude_patterns():
    """Test async_concat function with exclude_patterns."""
    # Create a temporary directory with some files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create files with different names
        file1_path = os.path.join(temp_dir, "file1.txt")
        file2_path = os.path.join(temp_dir, "file2.txt")
        file3_path = os.path.join(temp_dir, "temp_file.txt")
        
        with open(file1_path, "w") as f:
            f.write("Content of file 1")
        with open(file2_path, "w") as f:
            f.write("Content of file 2")
        with open(file3_path, "w") as f:
            f.write("Content of temp file")
        
        # Test concatenating files with exclude_patterns
        result = await async_concat(
            data_path=temp_dir,
            file_types=[".txt"],
            exclude_patterns=["temp"],  # Exclude files with "temp" in the name
            verbose=False,
        )
        
        assert "Content of file 1" in result["text"]
        assert "Content of file 2" in result["text"]
        assert "Content of temp file" not in result["text"]


@pytest.mark.asyncio
async def test_async_concat_return_files():
    """Test async_concat function with return_files=True."""
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
        result = await async_concat(
            data_path=[file1_path, file2_path],
            file_types=[".txt"],
            verbose=False,
            return_files=True,
        )
        
        assert "texts" in result
        assert isinstance(result["texts"], list)
        assert len(result["texts"]) > 0
        # The texts list contains file headers and content, so we check if any element contains our content
        assert any("Content of file 1" in text for text in result["texts"])
        assert any("Content of file 2" in text for text in result["texts"])


@pytest.mark.performance
async def test_sync_vs_async_concat_performance():
    """Test performance comparison between sync and async concat."""
    import asyncio
    import time
    from lionagi.libs.file.concat import concat

    # Create a temporary directory with multiple files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create 50 files with content
        file_paths = []
        for i in range(50):
            file_path = os.path.join(temp_dir, f"file{i}.txt")
            with open(file_path, "w") as f:
                f.write(f"Content of file {i}" * 100)  # Make it reasonably large
            file_paths.append(file_path)
        
        # Test synchronous concat
        start_time = time.time()
        concat(
            data_path=file_paths,
            file_types=[".txt"],
            verbose=False,
        )
        sync_time = time.time() - start_time
        
        # Test asynchronous concat
        start_time = time.time()
        await async_concat(
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