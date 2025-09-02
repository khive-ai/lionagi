#!/usr/bin/env python3
"""
Validate All .claude Files

Systematically validates all files in .claude directory for format compliance
"""

import re
from pathlib import Path
from typing import Any

import yaml

# Results tracking
validation_results = []
errors_found = []


def extract_yaml_from_markdown(content: str) -> dict[str, Any]:
    """Extract YAML blocks from markdown files"""
    yaml_blocks = re.findall(r"```yaml\n(.*?)\n```", content, re.DOTALL)
    if not yaml_blocks:
        return None

    try:
        # Parse the first YAML block
        return yaml.safe_load(yaml_blocks[0])
    except yaml.YAMLError as e:
        return {"error": f"YAML parsing error: {e}"}


def validate_event_file(file_path: str) -> tuple[bool, str]:
    """Validate event definition files"""
    try:
        with open(file_path) as f:
            content = f.read()

        yaml_data = extract_yaml_from_markdown(content)
        if not yaml_data:
            return False, "No YAML block found"

        if "error" in yaml_data:
            return False, yaml_data["error"]

        # Check required event fields
        required_fields = ["id", "name", "description", "triggers"]
        event_data = yaml_data.get("event", {})

        missing_fields = [
            field for field in required_fields if field not in event_data
        ]
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"

        # Validate event name format
        event_name = event_data.get("name", "")
        if not re.match(r"^[a-z]+\.[a-z_]+$", event_name):
            return False, f"Invalid event name format: {event_name}"

        # Validate ID format
        event_id = event_data.get("id", "")
        if not re.match(r"^\d{3}$", event_id):
            return (
                False,
                f"Invalid event ID format: {event_id} (should be 3 digits)",
            )

        return True, "Valid event definition"

    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_agent_file(file_path: str) -> tuple[bool, str]:
    """Validate agent specification files"""
    try:
        with open(file_path) as f:
            content = f.read()

        # Check required sections
        required_sections = ["Role", "Responsibilities"]
        for section in required_sections:
            if (
                f"## {section}" not in content
                and f"# {section}" not in content
            ):
                return False, f"Missing required section: {section}"

        # Check for any logic section (flexible naming)
        logic_sections = [
            "Decision Logic",
            "Analysis Logic",
            "Review Logic",
            "Tracking Logic",
            "Logic",
            "Processing Logic",
            "Analysis Patterns",
            "Discovery Strategies",
            "Planning Strategies",
            "Experiment Patterns",
            "Event-Driven Architecture",
            "Agent Delegation Protocol",
            "Performance Optimization",
            "Deduplication Strategies",
            "Metrics Framework",
            "Data Collection Patterns",
        ]
        has_logic_section = any(
            f"## {section}" in content for section in logic_sections
        )
        if not has_logic_section:
            return (
                False,
                "Missing logic section (Decision Logic, Analysis Logic, Review Logic, etc.)",
            )

        # Check for agent ID in filename
        filename = Path(file_path).stem
        if not filename.endswith("-agent"):
            return (
                False,
                f"Agent filename should end with '-agent': {filename}",
            )

        return True, "Valid agent specification"

    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_template_file(file_path: str) -> tuple[bool, str]:
    """Validate template files"""
    try:
        with open(file_path) as f:
            content = f.read()

        # Check for required template sections
        if "## Purpose" not in content:
            return False, "Missing Purpose section"

        # Skip YAML validation for GitHub template files (they contain markdown examples)
        if "github-templates" in str(file_path):
            return True, "Valid GitHub template format"

        if "## Template" not in content:
            return False, "Missing Template section"

        # Check for YAML block in data templates
        yaml_data = extract_yaml_from_markdown(content)
        if not yaml_data:
            return False, "No YAML template block found"

        if "error" in yaml_data:
            return False, yaml_data["error"]

        return True, "Valid template format"

    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_yaml_file(file_path: str) -> tuple[bool, str]:
    """Validate YAML configuration files"""
    try:
        with open(file_path) as f:
            yaml.safe_load(f)
        return True, "Valid YAML format"
    except yaml.YAMLError as e:
        return False, f"YAML syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_markdown_file(file_path: str) -> tuple[bool, str]:
    """Basic markdown file validation"""
    try:
        with open(file_path) as f:
            content = f.read()

        # Check for basic markdown structure
        if not content.strip():
            return False, "Empty file"

        # Check for at least one heading
        if not re.search(r"^#+ ", content, re.MULTILINE):
            return False, "No markdown headings found"

        return True, "Valid markdown format"

    except Exception as e:
        return False, f"Error reading file: {e}"


def main():
    claude_dir = Path(".claude")

    print("🔍 Validating all .claude files...")
    print("=" * 50)

    # Find all files
    all_files = []
    for ext in ["*.md", "*.yml", "*.yaml", "*.py"]:
        all_files.extend(claude_dir.rglob(ext))

    total_files = len(all_files)
    valid_files = 0

    for file_path in sorted(all_files):
        rel_path = file_path.relative_to(claude_dir)

        # Determine validation type based on path
        if "events/" in str(rel_path):
            is_valid, message = validate_event_file(file_path)
        elif "agents/" in str(rel_path):
            is_valid, message = validate_agent_file(file_path)
        elif "templates/" in str(rel_path) and rel_path.name.endswith(
            "-template.md"
        ):
            is_valid, message = validate_template_file(file_path)
        elif rel_path.suffix in [".yml", ".yaml"]:
            is_valid, message = validate_yaml_file(file_path)
        elif rel_path.suffix == ".md":
            is_valid, message = validate_markdown_file(file_path)
        else:
            is_valid, message = True, "Python file - skipping validation"

        # Record result
        status = "✅" if is_valid else "❌"
        print(f"{status} {rel_path}")
        if not is_valid:
            print(f"   💭 {message}")
            errors_found.append((str(rel_path), message))

        if is_valid:
            valid_files += 1

    print("\n" + "=" * 50)
    print("📊 Validation Summary:")
    print(f"   Total files: {total_files}")
    print(f"   Valid files: {valid_files}")
    print(f"   Invalid files: {total_files - valid_files}")

    if errors_found:
        print(f"\n❌ Errors found in {len(errors_found)} files:")
        for file_path, error in errors_found:
            print(f"   {file_path}: {error}")
        return 1
    print("\n🎉 All files validated successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
