from typing import Any, Literal

from pydantic import Field

from lionagi.fields.instruct import Instruct
from lionagi.models import BaseModel
from lionagi.utils import Enum


class TaskRequest(BaseModel):
    """Request for a specific task to be performed by an agent."""

    instruct: Instruct
    """The detailed instructions for the task to be performed by the agent."""

    role: str
    """The role of the agent performing the task"""

    domain: str
    """The domain or area of expertise for the agent"""


class OrchestrationPlan(BaseModel):
    """A plan for orchestrating agent tasks. Each plan is meant for either concurrent
    or sequential execution.
    """

    common_background: str
    """Common background information for all agents in the orchestration plan."""

    task_requests: list[TaskRequest]
    """List of task requests for each agent in the orchestration plan."""

    execution_strategy: Literal["sequential", "concurrent"] = "concurrent"
    """Execution strategy for the agents in the orchestration plan."""


class ComplexityAssessment(BaseModel):
    """Complexity assessment with overall score and explanation."""

    overall_complexity_score: float = Field(ge=0.0, le=1.0)
    """A normalized score between 0 and 1 representing the overall complexity of the task."""

    explanation: str
    """Detailed analysis of discovery findings, complexity factors, and rationale for score and path recommendation"""

    comment: str = Field(default_factory=str)
    """Additional context, risks, or implementation notes"""


class BaseGate(BaseModel):
    """Base quality gate with pass/fail and reasoning"""

    threshold_met: bool
    """You are evaluating work quality for a given task. Your assessment determines whether work can proceed to 
    the next stage or requires refinement. Be thorough and critical while being respectful of the task's required
    scope and effort (RETAIN FROM SCOPE CREEP), and ensure you consider all relevant criteria. 

    Only return `true` if ALL criteria are met and work is ready for next stage.
    """

    feedback: str | None = None
    """Provide constructive summary of what's working well and what needs improvement. Be specific and actionable in your 
    feedback. Common issue and mitigation strategies include:

    1. scope misalignment
    [problem] agents over-deliver, or under-deliver or fail to meet the task's requirements. 
    [mitigation] typically involves requirements refinement and re-evaluation, instruct  agents to edit, trim, or even remove 
        all changes and start from scratch if necessary.
        
    2. unverified or hallucinated information
    [problem] agents may include unverified, hallucinated, or incorrect information and claims in their responses.
    [mitigation] trust but verify. For testing and performances claims, run tests, benchmarks, or evaluations. for external 
        information, check against trusted sources, web searches, or databases. For project information, check against project 
        documentation, repositories, or other authoritative sources.
    """


class GateComponent(BaseModel):
    is_acceptable: bool
    """Is the relevant critiera met for the given gate component's requirements?"""

    problems: list[str] = Field(default_factory=list)
    """List specific problems or gaps that need addressing for this gate component"""


GateOptions = Literal[
    "design", "security", "performance", "testing", "documentation"
]


class FanoutPatterns(str, Enum):
    """Enumeration for orchestration patterns used in issues"""

    FANOUT = "fanout"
    W_REFINEMENT = "fanout_with_gated_refinement"
    COMPOSITE = "composite"


class FanoutConfig(BaseModel):
    initial_desc: str
    """Description for initial phase"""

    synth_instruction: str
    """Instruction for synthesis phase to generate deliverables"""

    planning_instruction: str
    """Instruction for planning phase to guide agent actions"""

    context: str | None = None
    """Context for the orchestration, can be None if not needed"""


class RefinementConfig(BaseModel):
    refinement_desc: str
    """Description for refinement phase if quality insufficient"""

    critic_domain: str = "software-architecture"
    """Domain for the critic to evaluate quality"""

    critic_role: str = "critic"
    """Role of the critic in the orchestration process"""

    gate_instruction: str
    """Instruction for the gate to evaluate quality of deliverables"""

    gates: Any = None
    """Optional gates for additional quality checks, can be a list or dict"""


class FanoutResponse(BaseModel):
    synth_node: Any | None = Field(None, exclude=True)
    """The synthesis node from the orchestration graph, if applicable"""

    synth_result: Any | None = None
    """The result from the synthesis node, if applicable"""

    flow_results: dict | None = Field(None, exclude=True)
    """The results from the flow execution, if applicable"""

    initial_nodes: list[Any] | None = Field(None, exclude=True)
    """The initial nodes from the orchestration graph, if applicable"""


class FanoutWithGatedRefinementResponse(FanoutResponse):
    final_gate: Any | None = Field(None, exclude=True)
    """The final gate node from the orchestration graph, if applicable"""

    qa_branch: Any | None = Field(None, exclude=True)
    """The quality assurance branch from the orchestration graph, if applicable"""

    gate_passed: bool | None = None
    """Whether the final gate was passed, if applicable"""

    refinement_executed: bool | None = None
    """Whether refinement was executed, if applicable"""


RefinementConfig.model_rebuild()
FanoutResponse.model_rebuild()
FanoutWithGatedRefinementResponse.model_rebuild()


__all__ = (
    "TaskRequest",
    "OrchestrationPlan",
    "ComplexityAssessment",
    "BaseGate",
    "GateComponent",
    "FanoutPatterns",
    "FanoutConfig",
    "RefinementConfig",
    "FanoutResponse",
    "FanoutWithGatedRefinementResponse",
)
