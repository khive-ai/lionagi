from typing import Any, Callable
from lionagi.models import FieldModel

from dataclasses import dataclass, field as dataclass_field
from .._concepts import Composable

@dataclass(slots=True, frozen=True, init=False)
class Operable(Composable):
    """
    Pydantic-free container for FieldModels representing cognitive output capabilities.
    
    Operable serves as a Composable invariant that defines what an intelligence 
    system can produce - the cognitive capabilities rather than system permissions.
    Field existence in validated output proves capability.
    
    Key Concepts:
    - Capability = cognitive ability to generate structured outputs
    - Container of FieldModel validation invariants
    - Bridges current OperableModel usage with IPU validation architecture
    - Enables trustless coordination through mathematical validation contracts
    
    Design Pattern:
    - Non-constructable: Use factory methods (.create(), .with_capability())
    - Aggressive caching following FieldModel patterns
    - Composition support for building sophisticated capability contracts
    - Materialization methods for lionagi integration
    
    Integration:
    - IPU validation: FrozenOperable provides immutable output contracts
    - lionagi compatibility: .to_field_model() maintains current patterns
    - CCP foundation: Implements mathematical capability validation
    """

    _field_models: dict[str, FieldModel] = dataclass_field(default_factory=dict)
    _name: str = dataclass_field(default="operable")
    _metadata: dict[str, Any] = dataclass_field(default_factory=dict)
    _version: str = dataclass_field(default="1.0.0")
    
    
    def __init__(
        self, 
                 field_models: dict[str, FieldModel] = None, 
                 name: str = "operable", 
                 metadata: Dict[str, Any] = None,
                 version: str = "1.0.0"):
        """
        Custom init following FieldModel pattern.
        
        IMPORTANT: This is not the public interface. Use factory methods:
        - Operable.create(name) for new instances
        - .with_capability() for adding capabilities
        - .with_metadata() for adding metadata
        """
        field_models = field_models or {}
        metadata = metadata or {}
        
        object.__setattr__(self, '_field_models', field_models.copy())
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_metadata', metadata.copy())
        object.__setattr__(self, '_version', version)
        
        self.__post_init__()
        
    def _get_cache_key(self) -> Tuple:
        """Generate cache key for this Operable instance"""
        return (
            "operable",
            self._name,
            tuple(sorted(self._field_models.keys())),
            tuple(sorted(self._metadata.items())),
            self._version
        )
        
    @classmethod
    def create(cls, name: str = "operable", version: str = "1.0.0") -> 'Operable':
        """
        Create new empty Operable capability container.
        
        Args:
            name: Descriptive name for this capability set
            version: Version for capability contract evolution
            
        Returns:
            New empty Operable ready for capability addition
        """
        cache_key = ("operable", name, (), (), version)
        return cls._get_cached_or_create(
            cache_key,
            lambda: cls({}, name, {}, version)
        )
        
        
    def with_capability(self, 
                       name: str, 
                       field_model: 'FieldModel' = None,
                       annotation: type = None,
                       description: str = None,
                       validator: Callable = None,
                       **kwargs) -> 'Operable':
        """
        Create new Operable with added cognitive capability.
        
        Args:
            name: Capability name (becomes field name in output)
            field_model: Existing FieldModel for this capability
            annotation: Type annotation if creating new FieldModel
            description: Human-readable capability description
            validator: Validation function for capability outputs
            **kwargs: Additional FieldModel parameters
            
        Returns:
            New Operable instance with added capability
            
        Example:
            research_ops = Operable.create("research_capabilities")
                .with_capability("summary", annotation=str, 
                               validator=lambda x: len(x) > 50)
                .with_capability("confidence", annotation=float,
                               validator=lambda x: 0 <= x <= 1)
        """
        if field_model is None:
            if annotation is None:
                raise ValueError("Must provide either field_model or annotation")
            
            # Import here to avoid circular imports
            from lionagi.models.field_model import FieldModel
            field_model = FieldModel(
                base_type=annotation,
                name=name,
                description=description,
                validator=validator,
                **kwargs
            )
        
        # Create new capability set
        new_field_models = {**self._field_models, name: field_model}
        cache_key = (
            "operable",
            self._name,
            tuple(sorted(new_field_models.keys())),
            tuple(sorted(self._metadata.items())),
            self._version
        )
        
        return self._get_cached_or_create(
            cache_key,
            lambda: type(self)(new_field_models, self._name, self._metadata, self._version)
        )
    
    def without_capability(self, name: str) -> 'Operable':
        """
        Create new Operable without specified capability.
        
        Args:
            name: Capability name to remove
            
        Returns:
            New Operable instance without the capability
        """
        if name not in self._field_models:
            return self  # No change needed, return same instance
        
        new_field_models = {k: v for k, v in self._field_models.items() if k != name}
        cache_key = (
            "operable",
            self._name,
            tuple(sorted(new_field_models.keys())),
            tuple(sorted(self._metadata.items())),
            self._version
        )
        
        return self._get_cached_or_create(
            cache_key,
            lambda: type(self)(new_field_models, self._name, self._metadata, self._version)
        )
    
    def with_name(self, name: str) -> 'Operable':
        """Create new Operable with different name"""
        if name == self._name:
            return self
        
        cache_key = (
            "operable",
            name,
            tuple(sorted(self._field_models.keys())),
            tuple(sorted(self._metadata.items())),
            self._version
        )
        
        return self._get_cached_or_create(
            cache_key,
            lambda: type(self)(self._field_models, name, self._metadata, self._version)
        )
    
    def with_metadata(self, key: str, value: Any) -> 'Operable':
        """Create new Operable with added metadata"""
        new_metadata = {**self._metadata, key: value}
        cache_key = (
            "operable",
            self._name,
            tuple(sorted(self._field_models.keys())),
            tuple(sorted(new_metadata.items())),
            self._version
        )
        
        return self._get_cached_or_create(
            cache_key,
            lambda: type(self)(self._field_models, self._name, new_metadata, self._version)
        )
    
    def with_version(self, version: str) -> 'Operable':
        """Create new Operable with updated version"""
        if version == self._version:
            return self
            
        cache_key = (
            "operable",
            self._name,
            tuple(sorted(self._field_models.keys())),
            tuple(sorted(self._metadata.items())),
            version
        )
        
        return self._get_cached_or_create(
            cache_key,
            lambda: type(self)(self._field_models, self._name, self._metadata, version)
        )
    
    # ---- Capability Query Methods ----
    
    def get_capability(self, name: str) -> Optional['FieldModel']:
        """Get a capability definition by name"""
        return self._field_models.get(name)
    
    def has_capability(self, name: str) -> bool:
        """Check if capability exists in this contract"""
        return name in self._field_models
    
    def list_capabilities(self) -> Set[str]:
        """List all capability names in this contract"""
        return set(self._field_models.keys())
    
    def get_capabilities_by_type(self, base_type: Type) -> Dict[str, 'FieldModel']:
        """Get all capabilities matching a specific type"""
        return {
            name: field_model 
            for name, field_model in self._field_models.items()
            if field_model.base_type == base_type
        }
    
    def get_validated_capabilities(self) -> Dict[str, 'FieldModel']:
        """Get all capabilities that have validators defined"""
        return {
            name: field_model
            for name, field_model in self._field_models.items()
            if hasattr(field_model, 'has_validator') and field_model.has_validator()
        }
    
    # ---- Composition Methods (Composable Interface) ----
    
    def merge_with(self, other: 'Operable', conflict_resolution: str = "error") -> 'Operable':
        """
        Merge capabilities from another Operable.
        
        Args:
            other: Another Operable to merge capabilities from
            conflict_resolution: How to handle capability name conflicts
                - "error": Raise error on conflicts (default)
                - "keep_first": Keep this Operable's capability
                - "take_second": Use other Operable's capability
                - "merge": Attempt to merge capability definitions
                
        Returns:
            New Operable with merged capabilities
        """
        if not isinstance(other, Operable):
            raise TypeError("Can only merge with another Operable")
        
        conflicts = set(self._field_models.keys()) & set(other._field_models.keys())
        
        if conflicts and conflict_resolution == "error":
            raise ValueError(f"Capability conflicts: {conflicts}")
        
        merged_field_models = self._field_models.copy()
        
        for name, field_model in other._field_models.items():
            if name in conflicts:
                if conflict_resolution == "keep_first":
                    continue  # Keep existing
                elif conflict_resolution == "take_second":
                    merged_field_models[name] = field_model
                elif conflict_resolution == "merge":
                    # Attempt capability merge (simplified - could be more sophisticated)
                    merged_field_models[name] = self._merge_field_models(
                        merged_field_models[name], field_model
                    )
            else:
                merged_field_models[name] = field_model
        
        # Create merged name
        merged_name = f"{self._name}_merged_{other._name}"
        
        cache_key = (
            "operable",
            merged_name,
            tuple(sorted(merged_field_models.keys())),
            tuple(sorted(self._metadata.items())),
            self._version
        )
        
        return self._get_cached_or_create(
            cache_key,
            lambda: type(self)(merged_field_models, merged_name, self._metadata, self._version)
        )
    
    def _merge_field_models(self, field1: 'FieldModel', field2: 'FieldModel') -> 'FieldModel':
        """Merge two FieldModel definitions (simplified implementation)"""
        # This is a simplified merge - real implementation might be more sophisticated
        return field1  # For now, just return first one
    
    # ---- Materialization Methods ----
    
    def generate_pydantic_model(self, model_name: str = None) -> Type:
        """
        Generate Pydantic model for lionagi compatibility.
        
        Args:
            model_name: Name for generated model class
            
        Returns:
            Dynamically created Pydantic model class
        """
        if not self._field_models:
            raise ValueError("No capabilities defined - cannot generate empty model")
        
        model_name = model_name or f"{self._name.title().replace('_', '')}Model"
        
        # Build field definitions for Pydantic model
        field_definitions = {}
        for name, field_model in self._field_models.items():
            # Use FieldModel's create_field method for Pydantic integration
            field_info = field_model.create_field()
            field_definitions[name] = (field_model.base_type, field_info)
        
        # Dynamically create Pydantic model
        try:
            from pydantic import create_model
            return create_model(model_name, **field_definitions)
        except ImportError:
            raise ImportError("Pydantic required for generate_pydantic_model()")
    
    def to_field_model(self, name: str = None) -> 'FieldModel':
        """
        Convert to FieldModel for lionagi operations (current compatibility).
        
        Args:
            name: Name for the resulting FieldModel
            
        Returns:
            FieldModel wrapping the generated Pydantic model
        """
        try:
            from lionagi.models.field_model import FieldModel
        except ImportError:
            raise ImportError("lionagi required for to_field_model()")
            
        pydantic_model = self.generate_pydantic_model()
        return FieldModel(
            base_type=pydantic_model,
            name=name or self._name,
            description=f"Generated from Operable '{self._name}' with {len(self._field_models)} capabilities"
        )
    
    def freeze(self) -> 'FrozenOperable':
        """
        Create immutable capability contract for IPU validation.
        
        Returns:
            FrozenOperable instance for use in IPU validation
        """
        return FrozenOperable.from_operable(self)
    
    
    # ---- Properties and Special Methods ----
    
    @property
    def name(self) -> str:
        """Capability contract name"""
        return self._name
    
    @property 
    def version(self) -> str:
        """Contract version for evolution tracking"""
        return self._version
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Contract metadata"""
        return self._metadata.copy()
    
    def __len__(self) -> int:
        """Number of capabilities in this contract"""
        return len(self._field_models)
    
    def __contains__(self, name: str) -> bool:
        """Check if capability exists"""
        return name in self._field_models
    
    def __iter__(self):
        """Iterate over capability names"""
        return iter(self._field_models)
    
    def __getitem__(self, name: str) -> 'FieldModel':
        """Get capability by name with dict-like access"""
        if name not in self._field_models:
            raise KeyError(f"Capability '{name}' not found in contract '{self._name}'")
        return self._field_models[name]
    
    def __repr__(self) -> str:
        return f"Operable('{self._name}', {len(self._field_models)} capabilities, v{self._version})"
    
    def __str__(self) -> str:
        capabilities = ", ".join(self._field_models.keys()) if self._field_models else "none"
        return f"Operable '{self._name}': [{capabilities}]"
    
    
    

class FrozenOperable:
    """
    Immutable version of Operable for IPU validation contracts.
    
    FrozenOperable represents a mathematical contract that cannot be modified,
    ensuring stable validation standards for IPU observation apparatus.
    Used as the measurement standard against which Observable outputs are validated.
    """
    
    __slots__ = ('_field_models', '_name', '_metadata', '_version', '_capabilities_hash')
    
    def __init__(self, 
                 field_models: Dict[str, 'FieldModel'], 
                 name: str, 
                 metadata: Dict[str, Any],
                 version: str = "1.0.0"):
        object.__setattr__(self, '_field_models', field_models)
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_metadata', metadata)
        object.__setattr__(self, '_version', version)
        object.__setattr__(self, '_capabilities_hash', 
                           hash(tuple(sorted(field_models.keys()))))
    
    @classmethod
    def from_operable(cls, operable: Operable) -> 'FrozenOperable':
        """Create FrozenOperable from mutable Operable"""
        return cls(
            field_models=operable._field_models.copy(),
            name=operable._name,
            metadata=operable._metadata.copy(),
            version=operable._version
        )
    
    # ---- Capability Query Methods ----
    
    def has_capability(self, name: str) -> bool:
        """Check if capability exists in this frozen contract"""
        return name in self._field_models
    
    def get_capability(self, name: str) -> Optional['FieldModel']:
        """Get capability definition"""
        return self._field_models.get(name)
    
    def list_capabilities(self) -> Set[str]:
        """List all capabilities in this frozen contract"""
        return set(self._field_models.keys())
    
    # ---- IPU Validation Methods ----
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate that output matches this capability contract.
        
        Core IPU validation method - determines if Observable output
        conforms to the mathematical standards defined in this contract.
        
        Args:
            output: Dictionary of capability name -> output value pairs
            
        Returns:
            True if output matches contract, False otherwise
        """
        # Check that output doesn't contain unexpected capabilities
        for field_name in output.keys():
            if not self.has_capability(field_name):
                return False
        
        # Validate each field using FieldModel validation
        for field_name, value in output.items():
            field_model = self.get_capability(field_name)
            if field_model and hasattr(field_model, 'is_valid'):
                if not field_model.is_valid(value):
                    return False
        
        return True
    
    def validate_capability_demonstration(self, capability: str, value: Any) -> bool:
        """
        Validate specific capability demonstration.
        
        Args:
            capability: Name of capability being demonstrated
            value: Output value for this capability
            
        Returns:
            True if value demonstrates the capability correctly
        """
        if not self.has_capability(capability):
            return False
        
        field_model = self.get_capability(capability)
        if field_model and hasattr(field_model, 'is_valid'):
            return field_model.is_valid(value)
        
        return True
    
    def get_validation_requirements(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed validation requirements for all capabilities.
        
        Returns:
            Dictionary mapping capability names to their validation requirements
        """
        requirements = {}
        for name, field_model in self._field_models.items():
            req = {
                "type": field_model.base_type,
                "required": True,  # All capabilities in contract are required
                "has_validator": hasattr(field_model, 'has_validator') and field_model.has_validator()
            }
            if hasattr(field_model, 'description') and field_model.description:
                req["description"] = field_model.description
            requirements[name] = req
        return requirements
    
    # ---- Materialization Methods ----
    
    def generate_pydantic_model(self, model_name: str = None) -> Type:
        """Generate Pydantic model from frozen capabilities"""
        model_name = model_name or f"Frozen{self._name.title().replace('_', '')}Model"
        
        field_definitions = {}
        for name, field_model in self._field_models.items():
            field_info = field_model.create_field()
            field_definitions[name] = (field_model.base_type, field_info)
        
        try:
            from pydantic import create_model
            return create_model(model_name, **field_definitions)
        except ImportError:
            raise ImportError("Pydantic required for generate_pydantic_model()")
    
    # ---- Properties and Special Methods ----
    
    @property
    def name(self) -> str:
        """Contract name"""
        return self._name
    
    @property
    def version(self) -> str:
        """Contract version"""
        return self._version
    
    @property
    def capabilities_hash(self) -> int:
        """Hash for capability contract identity"""
        return self._capabilities_hash
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Contract metadata (read-only)"""
        return self._metadata.copy()
    
    def __len__(self) -> int:
        return len(self._field_models)
    
    def __contains__(self, name: str) -> bool:
        return name in self._field_models
    
    def __iter__(self):
        return iter(self._field_models)
    
    def __getitem__(self, name: str) -> 'FieldModel':
        if name not in self._field_models:
            raise KeyError(f"Capability '{name}' not found in frozen contract '{self._name}'")
        return self._field_models[name]
    
    def __hash__(self) -> int:
        return self._capabilities_hash
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, FrozenOperable):
            return False
        return self._capabilities_hash == other._capabilities_hash
    
    def __repr__(self) -> str:
        return f"FrozenOperable('{self._name}', {len(self._field_models)} capabilities, v{self._version})"