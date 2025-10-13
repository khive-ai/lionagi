"""Usage examples for LSpec v2.

Run with: python -m lionagi.specs_v2.examples
"""

from lionagi.specs_v2 import (
    FieldSpec,
    BackendRegistry,
    PydanticBackend,
    # RustBackend,  # Requires lionagi[rust]
    # CloudBackend,  # Requires API key
)


def example_basic_validation():
    """Example 1: Basic validation with Pydantic backend."""
    print("\n=== Example 1: Basic Validation ===\n")

    # Register Pydantic backend
    BackendRegistry.register("pydantic", PydanticBackend())
    BackendRegistry.set_default("pydantic")

    # Define field specs
    age_spec = FieldSpec(int, {"min": 0, "max": 120})
    email_spec = FieldSpec(str, {"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"})

    # Validate valid values
    print("Validating age=25...")
    age = BackendRegistry.validate(age_spec, 25)
    print(f"✓ Valid: {age}")

    print("\nValidating email='user@example.com'...")
    email = BackendRegistry.validate(email_spec, "user@example.com")
    print(f"✓ Valid: {email}")

    # Try invalid values
    print("\nValidating age=150 (should fail)...")
    try:
        BackendRegistry.validate(age_spec, 150)
    except Exception as e:
        print(f"✗ Invalid: {e}")


def example_composition():
    """Example 2: Field composition and transformations."""
    print("\n=== Example 2: Field Composition ===\n")

    # Base instruction field
    base_instruct = FieldSpec(str, {"description": "Task to perform", "min_length": 1})
    print(f"Base spec: {base_instruct}")

    # Make nullable
    nullable_instruct = base_instruct.as_nullable()
    print(f"Nullable spec: {nullable_instruct}")

    # Make listable
    list_instruct = base_instruct.as_listable()
    print(f"List spec: {list_instruct}")

    # Chain transformations
    optional_list = base_instruct.as_listable().as_nullable()
    print(f"Optional list spec: {optional_list}")


def example_serialization():
    """Example 3: Serialization for backend communication."""
    print("\n=== Example 3: Serialization ===\n")

    # Create a spec
    spec = FieldSpec(int, {"min": 0, "max": 100, "description": "Score"})
    print(f"Original spec: {spec}")

    # Serialize to dict
    spec_dict = spec.to_dict()
    print(f"\nSerialized: {spec_dict}")

    # Deserialize back
    restored = FieldSpec.from_dict(spec_dict)
    print(f"\nRestored: {restored}")
    print(f"Equal: {spec.type == restored.type and spec.constraints == restored.constraints}")


def example_reusable_specs():
    """Example 4: Reusable field specifications."""
    print("\n=== Example 4: Reusable Specs ===\n")

    # Define reusable specs (like constants)
    AGE_SPEC = FieldSpec(int, {"min": 0, "max": 120})
    EMAIL_SPEC = FieldSpec(str, {"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"})
    USERNAME_SPEC = FieldSpec(str, {"min_length": 3, "max_length": 20, "pattern": r"^[a-zA-Z0-9_]+$"})

    # Use across multiple models/contexts
    print("User profile validation:")
    user_data = {
        "age": 25,
        "email": "user@example.com",
        "username": "john_doe"
    }

    validated = {}
    validated["age"] = BackendRegistry.validate(AGE_SPEC, user_data["age"])
    validated["email"] = BackendRegistry.validate(EMAIL_SPEC, user_data["email"])
    validated["username"] = BackendRegistry.validate(USERNAME_SPEC, user_data["username"])

    print(f"✓ Validated user: {validated}")


def example_backend_switching():
    """Example 5: Switching between backends."""
    print("\n=== Example 5: Backend Switching ===\n")

    # Register multiple backends
    BackendRegistry.register("pydantic", PydanticBackend())

    spec = FieldSpec(int, {"min": 0, "max": 100})
    value = 42

    # Validate with free tier (Pydantic)
    print("Validating with Pydantic (free tier)...")
    result = BackendRegistry.validate(spec, value, backend="pydantic")
    print(f"✓ Valid (Pydantic): {result}")

    # Try Rust backend (will show stub message)
    print("\nValidating with Rust (paid tier)...")
    try:
        # This would work if Rust backend was installed
        # from lionagi.specs_v2 import RustBackend
        # BackendRegistry.register("rust", RustBackend())
        # result = BackendRegistry.validate(spec, value, backend="rust")
        # print(f"✓ Valid (Rust): {result}")
        print("(Rust backend not installed - would provide formal verification)")
    except Exception as e:
        print(f"Note: {e}")

    # Try Cloud backend (would require API key)
    print("\nValidating with Cloud (enterprise tier)...")
    try:
        # This would work with API key
        # from lionagi.specs_v2 import CloudBackend
        # cloud = CloudBackend(api_key="...")
        # BackendRegistry.register("cloud", cloud)
        # result = BackendRegistry.validate(spec, value, backend="cloud")
        print("(Cloud backend not configured - would provide guarantees + audit trail)")
    except Exception as e:
        print(f"Note: {e}")


def example_custom_constraints():
    """Example 6: Adding custom constraints."""
    print("\n=== Example 6: Custom Constraints ===\n")

    # Start with base spec
    spec = FieldSpec(int, {})
    print(f"Base: {spec}")

    # Add constraints one by one
    spec = spec.with_constraint("min", 0)
    print(f"With min: {spec}")

    spec = spec.with_constraint("max", 100)
    print(f"With max: {spec}")

    spec = spec.with_constraint("description", "A score between 0 and 100")
    print(f"With description: {spec}")

    # Updating constraint (replaces old value)
    spec = spec.with_constraint("max", 200)  # Replaces max=100 with max=200
    print(f"Updated max: {spec}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("LSpec v2 Examples")
    print("=" * 60)

    example_basic_validation()
    example_composition()
    example_serialization()
    example_reusable_specs()
    example_backend_switching()
    example_custom_constraints()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
