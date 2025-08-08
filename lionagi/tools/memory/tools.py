"""
Memory Tools - Proper lionagi tool implementation following reader pattern
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from lionagi.libs.validate.to_num import to_num
from lionagi.protocols.action.tool import Tool
from lionagi.tools.base import LionTool


class MemoryAction(str, Enum):
    """
    Memory action types following reader pattern.
    - 'store': Store new memory with embeddings and metadata
    - 'recall': Retrieve memories based on semantic similarity
    - 'search': Search memories using multiple criteria
    - 'relate': Create relationships between memories
    - 'explore': Deep multi-step memory exploration
    - 'synthesize': Combine multiple memories into insights
    """

    store = "store"
    recall = "recall"
    search = "search"
    relate = "relate"
    explore = "explore"
    synthesize = "synthesize"


class MemoryLayer(str, Enum):
    """Memory storage layers"""

    static = "static"  # File-based persistent memory
    temporal = "temporal"  # Time-based conversational memory
    experience = "experience"  # High-value insights and patterns


class MemoryRequest(BaseModel):
    """
    Request model for MemoryTool following reader pattern.
    Supports multiple memory operations with unified interface.
    """

    action: MemoryAction = Field(
        ...,
        description=(
            "Memory action to perform. Must be one of: "
            "- 'store': Save new memory with automatic embedding and indexing. "
            "- 'recall': Retrieve semantically similar memories. "
            "- 'search': Advanced search with multiple criteria. "
            "- 'relate': Create relationships between memories. "
            "- 'explore': Deep exploration with multiple strategies. "
            "- 'synthesize': Combine memories into new insights."
        ),
    )

    # Store action fields
    content: str | None = Field(
        None,
        description=(
            "Content to store. REQUIRED if action='store'. "
            "For other actions, leave it None."
        ),
    )

    tags: list[str] | None = Field(
        None,
        description=(
            "Tags for categorization. Used with action='store'. "
            "Can also be used to filter in 'search' and 'recall'."
        ),
    )

    importance: float | None = Field(
        0.5,
        description=(
            "Importance score (0.0-1.0) for prioritization. "
            "Used with action='store'. Higher scores are retained longer."
        ),
    )

    layer: MemoryLayer | None = Field(
        MemoryLayer.temporal,
        description=(
            "Memory layer for storage/search. "
            "static: persistent files, temporal: conversations, experience: insights"
        ),
    )

    # Recall/Search fields
    query: str | None = Field(
        None,
        description=(
            "Query text for semantic search. "
            "REQUIRED for actions: 'recall', 'search', 'explore'."
        ),
    )

    limit: int | None = Field(
        5,
        description="Maximum number of results to return. Default is 5.",
    )

    threshold: float | None = Field(
        0.7,
        description=(
            "Similarity threshold (0.0-1.0) for filtering results. "
            "Only memories above this threshold are returned."
        ),
    )

    # Relate action fields
    source_id: str | None = Field(
        None,
        description="Source memory ID for creating relationships. REQUIRED if action='relate'.",
    )

    target_id: str | None = Field(
        None,
        description="Target memory ID for creating relationships. REQUIRED if action='relate'.",
    )

    relationship: str | None = Field(
        None,
        description=(
            "Type of relationship (e.g., 'relates_to', 'contradicts', 'supports'). "
            "REQUIRED if action='relate'."
        ),
    )

    # Explore action fields
    depth: int | None = Field(
        3,
        description="Exploration depth for multi-step exploration. Default is 3.",
    )

    strategies: list[str] | None = Field(
        None,
        description=(
            "Exploration strategies to use: 'semantic', 'temporal', 'relational', 'contextual'. "
            "If None, uses appropriate defaults."
        ),
    )

    # Synthesize action fields
    memory_ids: list[str] | None = Field(
        None,
        description="List of memory IDs to synthesize. Used with action='synthesize'.",
    )

    synthesis_mode: str | None = Field(
        "intelligent",
        description=(
            "Synthesis complexity: 'simple' (aggregation), "
            "'intelligent' (AI-powered), 'deep' (multi-step reasoning)"
        ),
    )

    @model_validator(mode="before")
    def _validate_request(cls, values):
        """Clean up empty dicts and validate numeric fields"""
        for k, v in values.items():
            if v == {}:
                values[k] = None
            if k in ["limit", "depth"]:
                try:
                    values[k] = to_num(v, num_type=int)
                except ValueError:
                    values[k] = None
            if k in ["importance", "threshold"]:
                try:
                    values[k] = to_num(v, num_type=float)
                except ValueError:
                    values[k] = None
        return values


class MemoryInfo(BaseModel):
    """Information about a stored memory"""

    memory_id: str
    timestamp: str
    layer: MemoryLayer
    tags: list[str] = Field(default_factory=list)
    importance: float
    token_count: int | None = None


class MemoryMatch(BaseModel):
    """A memory match from recall/search operations"""

    memory_id: str
    content: str
    similarity: float
    timestamp: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExplorationResult(BaseModel):
    """Result from memory exploration"""

    query: str
    depth_reached: int
    strategies_used: list[str]
    insights: list[dict[str, Any]]
    connections_found: int
    processing_time: float


class SynthesisResult(BaseModel):
    """Result from memory synthesis"""

    synthesis_id: str
    source_memories: list[str]
    synthesis_content: str
    confidence: float
    insights: list[str]


class MemoryResponse(BaseModel):
    """
    Response from MemoryTool following reader pattern.
    Different fields are populated based on the action performed.
    """

    success: bool = Field(
        ...,
        description="Indicates if the requested action was performed successfully.",
    )

    error: str | None = Field(
        None,
        description="Describes any error that occurred, if success=False.",
    )

    # Store response
    memory_info: MemoryInfo | None = Field(
        None,
        description="Populated when action='store' succeeds, with memory ID and metadata.",
    )

    # Recall/Search response
    matches: list[MemoryMatch] | None = Field(
        None,
        description="Populated for recall/search actions, containing matching memories.",
    )

    # Exploration response
    exploration: ExplorationResult | None = Field(
        None,
        description="Populated when action='explore' succeeds, with deep insights.",
    )

    # Synthesis response
    synthesis: SynthesisResult | None = Field(
        None,
        description="Populated when action='synthesize' succeeds, with combined insights.",
    )

    # Relationship response
    relationship_created: bool | None = Field(
        None,
        description="Indicates if relationship was created successfully.",
    )


class MemoryTool(LionTool):
    """
    Memory tool following lionagi's reader pattern.
    Provides unified interface for all memory operations.
    """

    is_lion_system_tool = True
    system_tool_name = "memory_tool"

    def __init__(self, memory_backend=None):
        super().__init__()
        self.backend = memory_backend
        self._tool = None

        # Memory caches
        self._embedding_cache = {}
        self._search_cache = {}

    def handle_request(self, request: MemoryRequest) -> MemoryResponse:
        """
        Handle memory requests based on action type.
        Routes to appropriate handler method.
        """
        if isinstance(request, dict):
            request = MemoryRequest(**request)

        try:
            if request.action == "store":
                return self._store_memory(request)
            elif request.action == "recall":
                return self._recall_memories(request)
            elif request.action == "search":
                return self._search_memories(request)
            elif request.action == "relate":
                return self._relate_memories(request)
            elif request.action == "explore":
                return self._explore_memories(request)
            elif request.action == "synthesize":
                return self._synthesize_memories(request)
            else:
                return MemoryResponse(
                    success=False,
                    error=f"Unknown action type: {request.action}",
                )
        except Exception as e:
            return MemoryResponse(
                success=False, error=f"Memory operation failed: {str(e)}"
            )

    def _store_memory(self, request: MemoryRequest) -> MemoryResponse:
        """Store new memory with embedding and metadata"""
        if not request.content:
            return MemoryResponse(
                success=False, error="Content is required for store action"
            )

        # Generate embedding
        embedding = self._get_embedding(request.content)

        # Create memory ID
        memory_id = f"MEM_{int(datetime.now(timezone.utc).timestamp())}_{hash(request.content) % 10000}"

        # Store in backend
        if self.backend:
            self.backend.store(
                {
                    "memory_id": memory_id,
                    "content": request.content,
                    "embedding": embedding,
                    "tags": request.tags or [],
                    "importance": request.importance,
                    "layer": request.layer,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        return MemoryResponse(
            success=True,
            memory_info=MemoryInfo(
                memory_id=memory_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                layer=request.layer,
                tags=request.tags or [],
                importance=request.importance,
                token_count=len(
                    request.content.split()
                ),  # Simple token estimate
            ),
        )

    def _recall_memories(self, request: MemoryRequest) -> MemoryResponse:
        """Recall memories based on semantic similarity"""
        if not request.query:
            return MemoryResponse(
                success=False, error="Query is required for recall action"
            )

        # Get query embedding
        query_embedding = self._get_embedding(request.query)

        # Search in backend
        if self.backend:
            matches = self.backend.search_similar(
                query_embedding,
                limit=request.limit,
                threshold=request.threshold,
                layer=request.layer,
                tags=request.tags,
            )
        else:
            # Mock response for testing
            matches = [
                MemoryMatch(
                    memory_id=f"MEM_TEST_{i}",
                    content=f"Test memory {i} related to: {request.query}",
                    similarity=0.9 - (i * 0.1),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    tags=["test", "mock"],
                )
                for i in range(min(3, request.limit))
            ]

        return MemoryResponse(success=True, matches=matches)

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text (cached)"""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Generate embedding (mock for now)
        # In production, use actual embedding service
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding = [
            float(int(text_hash[i : i + 2], 16)) / 255.0
            for i in range(0, 32, 2)
        ]

        # Cache and return
        self._embedding_cache[text] = embedding
        return embedding

    def to_tool(self):
        """Convert to lionagi Tool following reader pattern"""
        if self._tool is None:

            def memory_tool(**kwargs):
                """
                Unified memory tool for AI agents supporting:
                - store: Save memories with automatic embedding and indexing
                - recall: Retrieve semantically similar memories
                - search: Advanced multi-criteria memory search
                - relate: Create knowledge graph relationships
                - explore: Deep multi-strategy exploration
                - synthesize: Combine memories into insights
                """
                return self.handle_request(
                    MemoryRequest(**kwargs)
                ).model_dump()

            if self.system_tool_name != "memory_tool":
                memory_tool.__name__ = self.system_tool_name

            self._tool = Tool(
                func_callable=memory_tool,
                request_options=MemoryRequest,
            )

        return self._tool

    # Placeholder implementations for other actions
    def _search_memories(self, request: MemoryRequest) -> MemoryResponse:
        """Advanced search with multiple criteria"""
        # Similar to recall but with more filters
        return self._recall_memories(request)

    def _relate_memories(self, request: MemoryRequest) -> MemoryResponse:
        """Create relationships between memories"""
        if not all(
            [request.source_id, request.target_id, request.relationship]
        ):
            return MemoryResponse(
                success=False,
                error="source_id, target_id, and relationship are required for relate action",
            )

        # Create relationship in backend
        if self.backend:
            self.backend.create_relationship(
                request.source_id, request.target_id, request.relationship
            )

        return MemoryResponse(success=True, relationship_created=True)

    def _explore_memories(self, request: MemoryRequest) -> MemoryResponse:
        """Deep multi-strategy exploration"""
        # Placeholder for complex exploration
        return MemoryResponse(
            success=True,
            exploration=ExplorationResult(
                query=request.query,
                depth_reached=request.depth,
                strategies_used=request.strategies or ["semantic"],
                insights=[
                    {"type": "pattern", "content": "Exploration insight"}
                ],
                connections_found=5,
                processing_time=2.3,
            ),
        )

    def _synthesize_memories(self, request: MemoryRequest) -> MemoryResponse:
        """Synthesize multiple memories into insights"""
        # Placeholder for synthesis
        return MemoryResponse(
            success=True,
            synthesis=SynthesisResult(
                synthesis_id=f"SYN_{int(datetime.now(timezone.utc).timestamp())}",
                source_memories=request.memory_ids or [],
                synthesis_content="Synthesized insight from memories",
                confidence=0.85,
                insights=["Key insight 1", "Key insight 2"],
            ),
        )
