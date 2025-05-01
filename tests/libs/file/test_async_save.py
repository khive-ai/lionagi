import json
import os
import tempfile
from pathlib import Path

import pytest

from lionagi.libs.file.save import async_save_chunks, async_save_to_file


@pytest.mark.asyncio
async def test_async_save_to_file():
    """Test async_save_to_file function."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test saving a file
        text = "Test content"
        file_path = await async_save_to_file(
            text=text,
            directory=temp_dir,
            filename="test_file",
            extension="txt",
            verbose=False,
        )

        assert file_path.exists()
        with open(file_path, "r") as f:
            assert f.read() == text

        # Test saving with timestamp
        file_path = await async_save_to_file(
            text=text,
            directory=temp_dir,
            filename="test_file_timestamp",
            extension="txt",
            timestamp=True,
            verbose=False,
        )

        assert file_path.exists()
        # Filename should contain the timestamp
        assert file_path.stem != "test_file_timestamp"
        assert "test_file_timestamp" in file_path.stem
        with open(file_path, "r") as f:
            assert f.read() == text

        # Test saving with random hash
        file_path = await async_save_to_file(
            text=text,
            directory=temp_dir,
            filename="test_file_hash",
            extension="txt",
            random_hash_digits=6,
            verbose=False,
        )

        assert file_path.exists()
        # Filename should contain the hash
        assert file_path.stem != "test_file_hash"
        assert "test_file_hash" in file_path.stem
        with open(file_path, "r") as f:
            assert f.read() == text


@pytest.mark.asyncio
async def test_async_save_chunks():
    """Test async_save_chunks function."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test chunks
        chunks = [
            {"chunk_content": "Chunk 1", "chunk_id": 1},
            {"chunk_content": "Chunk 2", "chunk_id": 2},
        ]

        # Test saving chunks
        await async_save_chunks(
            chunks=chunks,
            output_dir=temp_dir,
            verbose=False,
            timestamp=False,
            random_hash_digits=0,
        )

        # Verify the chunks were saved
        files = list(Path(temp_dir).glob("*.json"))
        assert len(files) == 2

        # Verify the content of the files
        for i, file in enumerate(sorted(files, key=lambda x: x.name)):
            with open(file, "r") as f:
                content = json.load(f)
                assert content["chunk_id"] == i + 1
                assert content["chunk_content"] == f"Chunk {i + 1}"

        # Test saving chunks with timestamp
        with tempfile.TemporaryDirectory() as temp_dir2:
            await async_save_chunks(
                chunks=chunks,
                output_dir=temp_dir2,
                verbose=False,
                timestamp=True,
                random_hash_digits=0,
            )

            # Verify the chunks were saved
            files = list(Path(temp_dir2).glob("*.json"))
            assert len(files) == 2

            # Verify the filenames contain timestamps
            for file in files:
                assert "chunk_" in file.stem
                # The filename should be longer than just "chunk_N" due to timestamp
                assert len(file.stem) > 8


@pytest.mark.performance
async def test_sync_vs_async_save_performance():
    """Test performance comparison between sync and async file saving."""
    import asyncio
    import time

    from lionagi.libs.file.save import save_to_file

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test content
        num_files = 50
        text = "Test content" * 1000  # Make it reasonably large

        # Test synchronous saving
        start_time = time.time()
        for i in range(num_files):
            save_to_file(
                text=text,
                directory=temp_dir,
                filename=f"sync_file_{i}",
                extension="txt",
                verbose=False,
            )
        sync_time = time.time() - start_time

        # Test asynchronous saving
        start_time = time.time()
        tasks = [
            async_save_to_file(
                text=text,
                directory=temp_dir,
                filename=f"async_file_{i}",
                extension="txt",
                verbose=False,
            )
            for i in range(num_files)
        ]
        await asyncio.gather(*tasks)
        async_time = time.time() - start_time

        # For small files or in certain environments, async might have more overhead
        # We're just printing the times for information, not asserting
        print(f"Sync time: {sync_time:.4f}s, Async time: {async_time:.4f}s")

        # The real benefit of async I/O is seen with many concurrent operations
        # or when I/O operations are slow (like network requests)
        # For local file operations, the difference might not be as pronounced


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
