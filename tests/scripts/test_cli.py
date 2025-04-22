import os
import sys
import tempfile
from unittest.mock import patch

import pytest

from lionagi.scripts.cli import main


def test_cli_build_registry():
    """Test the CLI build-registry command."""
    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "adapter_map.json")
        
        # Create a temporary directory with adapter files
        with tempfile.TemporaryDirectory() as adapters_dir:
            # Create an adapter file
            with open(os.path.join(adapters_dir, "test_adapter.py"), "w") as f:
                f.write(
                    """
class TestAdapter:
    obj_key = "test"
"""
                )
            
            # Mock the command-line arguments
            with patch("sys.argv", ["lionagi", "build-registry", "--adapters-dir", adapters_dir, "--output", output_file]):
                # Mock the build_registry_main function to verify it's called with the correct arguments
                with patch("lionagi.scripts.build_adapter_registry.main") as mock_build_registry:
                    # Run the CLI
                    main()
                    
                    # Verify the build_registry_main function was called
                    assert mock_build_registry.called