#!/usr/bin/env python
# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Build hooks for hatch to generate the adapter registry during build.
"""

import logging
import os
import sys
from pathlib import Path

from hatchling.builders.hooks.interface import BuildHookInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build-hooks")


class AdapterRegistryBuildHook(BuildHookInterface):
    """
    Build hook to generate the adapter registry during build.
    """
    
    def initialize(self, version, build_data):
        """
        Initialize hook - called before the build process starts.
        
        Args:
            version: The version of the package being built
            build_data: Build data dictionary
        """
        logger.info(f"Initializing build hooks for lionagi v{version}")
        return build_data

    def clean(self, versions, build_data):
        """
        Clean hook - called before cleaning build artifacts.
        
        Args:
            versions: List of versions to clean
            build_data: Build data dictionary
        """
        logger.info(f"Cleaning build artifacts for versions: {versions}")
        return build_data

    def build_sdist(self, sdist_directory, build_data):
        """
        Build sdist hook - called when building source distributions.
        
        Args:
            sdist_directory: Directory where the sdist will be created
            build_data: Build data dictionary
        """
        logger.info(f"Building source distribution in {sdist_directory}")
        
        # Generate the adapter registry
        self._generate_adapter_registry()
        
        return build_data

    def build_wheel(self, wheel_directory, build_data):
        """
        Build wheel hook - called when building wheel distributions.
        
        Args:
            wheel_directory: Directory where the wheel will be created
            build_data: Build data dictionary
        """
        logger.info(f"Building wheel in {wheel_directory}")
        
        # Generate the adapter registry
        self._generate_adapter_registry()
        
        return build_data

    def finalize(self, build_data):
        """
        Finalize hook - called after the build process completes.
        
        Args:
            build_data: Build data dictionary
        """
        logger.info("Finalizing build")
        return build_data

    def _generate_adapter_registry(self):
        """
        Generate the adapter registry by running the build_adapter_registry script.
        """
        try:
            logger.info("Generating adapter registry...")
            
            # Import the build_adapter_registry module
            from lionagi.scripts.build_adapter_registry import main as build_registry_main
            
            # Run the build_adapter_registry script
            build_registry_main()
            
            logger.info("Adapter registry generated successfully")
        except Exception as e:
            logger.error(f"Error generating adapter registry: {e}")
            # Don't fail the build if the registry generation fails
            # This ensures the package can still be installed
            logger.warning("Continuing build despite adapter registry generation failure")


# Create an instance of the build hook for hatch to use
build_hook = AdapterRegistryBuildHook