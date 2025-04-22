#!/usr/bin/env python
# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
CLI script to pre-compute the adapter registry by scanning the lionagi/adapters directory
and generating a static mapping of adapter keys to their module paths.

This script is intended to be run during the build process to generate a static mapping
that can be loaded at runtime, avoiding the need for expensive filesystem scans.
"""

import argparse
import ast
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build-adapter-registry")


def is_adapter_class(node: ast.ClassDef) -> bool:
    """
    Check if a class definition is an adapter class by looking for the obj_key attribute.
    
    Args:
        node: The AST node representing a class definition
        
    Returns:
        bool: True if the class has an obj_key attribute, False otherwise
    """
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "obj_key":
                    return True
    return False


def extract_obj_key(node: ast.ClassDef) -> Optional[str]:
    """
    Extract the obj_key value from a class definition.
    
    Args:
        node: The AST node representing a class definition
        
    Returns:
        Optional[str]: The obj_key value if found, None otherwise
    """
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "obj_key":
                    if isinstance(item.value, ast.Constant):
                        return item.value.value
    return None


def find_adapter_classes(file_path: str) -> List[Tuple[str, str]]:
    """
    Find adapter classes in a Python file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        List[Tuple[str, str]]: List of (class_name, obj_key) tuples
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        
        tree = ast.parse(file_content)
        adapter_classes = []
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and is_adapter_class(node):
                obj_key = extract_obj_key(node)
                if obj_key is not None:
                    adapter_classes.append((node.name, obj_key))
        
        return adapter_classes
    except Exception as e:
        logger.warning(f"Error parsing {file_path}: {e}")
        return []


def scan_adapters_directory(directory: str) -> Dict[str, str]:
    """
    Scan the adapters directory for adapter classes and build a mapping of
    obj_key to module path.
    
    Args:
        directory: Path to the adapters directory
        
    Returns:
        Dict[str, str]: Mapping of obj_key to module path
    """
    adapter_map = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, os.path.dirname(directory))
                module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
                
                adapter_classes = find_adapter_classes(file_path)
                for class_name, obj_key in adapter_classes:
                    full_class_path = f"{module_path}.{class_name}"
                    adapter_map[obj_key] = full_class_path
                    logger.info(f"Found adapter: {obj_key} -> {full_class_path}")
    
    return adapter_map


def main():
    parser = argparse.ArgumentParser(
        description="Build a static mapping of adapter keys to module paths."
    )
    parser.add_argument(
        "--adapters-dir",
        type=str,
        default=None,
        help="Path to the adapters directory (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output JSON file (default: adapter_map.json in the package directory)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Auto-detect the adapters directory if not specified
    if args.adapters_dir is None:
        # Find the lionagi package directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(script_dir)
        adapters_dir = os.path.join(package_dir, "adapters")
    else:
        adapters_dir = args.adapters_dir
    
    # Auto-detect the output path if not specified
    if args.output is None:
        # Use the package directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(script_dir)
        output_path = os.path.join(package_dir, "adapter_map.json")
    else:
        output_path = args.output
    
    # Ensure the adapters directory exists
    if not os.path.isdir(adapters_dir):
        logger.error(f"Adapters directory not found: {adapters_dir}")
        sys.exit(1)
    
    logger.info(f"Scanning adapters directory: {adapters_dir}")
    adapter_map = scan_adapters_directory(adapters_dir)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the adapter map to the output file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(adapter_map, f, indent=2)
    
    logger.info(f"Adapter map written to: {output_path}")
    logger.info(f"Found {len(adapter_map)} adapters")


if __name__ == "__main__":
    main()