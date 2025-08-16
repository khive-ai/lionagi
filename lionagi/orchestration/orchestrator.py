import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal

import aiofiles

from lionagi import Branch, Builder, Operation, Session, iModel
from lionagi.fields import Instruct
from lionagi.models import FieldModel, OperableModel
from lionagi.protocols.types import ID, AssistantResponse, Graph, IDType, Pile

from .parts import (
    BaseGate,
    FanoutResponse,
    FanoutWithGatedRefinementResponse,
    GateComponent,
    GateOptions,
    OrchestrationPlan,
    TaskRequest,
)
from .prompts.gates import get_gate_prompt, list_available_gates

logger = logging.getLogger("lionagi.orchestration")


class LionOrchestrator:

    def __init__(self, name, max_concurrent: int = 8, user=None):
        self.name = name
        self.session = None
        self.builder = None
        self.max_concurrent = max_concurrent
        self.user = user or "default_user"

    async def initialize(self, model: str = None, system: str = None, **kw):
        orc_model = await self.create_orc_imodel(model=model, **kw)
        orc_branch = Branch(
            chat_model=orc_model,
            parse_model=orc_model,
            system=system,
            use_lion_system_message=True,
            system_datetime=True,
        )
        self.session = Session(
            default_branch=orc_branch, name=self.name, user=self.user
        )
        self.builder = Builder(self.name)

    async def run_flow(self, visualize: bool = False):
        """Run flow with timeout protection and security logging."""
        if visualize:
            self.builder.visualize(self.session.name)
        result = await self.session.flow(
            self.builder.get_graph(), max_concurrent=self.max_concurrent
        )
        return result

    @property
    def orc_branch(self):
        return self.session.default_branch

    @staticmethod
    def generate_flow_plans_field(**plans_description: str):
        from .parts import OrchestrationPlan

        a = OperableModel()
        for plan, doc in plans_description.items():
            a.add_field(plan, annotation=OrchestrationPlan, description=doc)

        return FieldModel(
            base_type=a.new_model("FlowOrchestrationPlans"), name="flow_plans"
        )

    @staticmethod
    def generate_quality_gate_field(**gate_components):
        op = OperableModel()
        gates = list_available_gates()

        for k, doc in gate_components.items():
            if not isinstance(doc, str):
                if k not in gates:
                    raise ValueError(
                        f"Unknown gate component '{k}'. Available gates: {', '.join(gates)}"
                    )
                doc = get_gate_prompt(k)
            if doc:
                op.add_field(k, annotation=GateComponent, description=doc)

        return FieldModel(
            base_type=op.new_model("QualityGate", base_type=BaseGate),
            name="quality_gate",
        )

    @abstractmethod
    async def create_agent_imodel(self, *, role, domain, **kw) -> iModel:
        pass

    @abstractmethod
    async def create_orc_imodel(self, **kw) -> iModel:
        pass

    async def expand_with_plan(
        self,
        root: IDType | list[IDType],
        plan: OrchestrationPlan,
        max_agents: int = 8,
        auto_context: bool = True,
        skip_root_context: bool = True,
        **kw,
    ) -> list[ID[Operation]]:

        nodes = []
        root_set = {root} if isinstance(root, IDType) else set(root)
        dep_on = [root] if not isinstance(root, list) else root

        _ctx = {"step_ctx": {}}
        if auto_context and not skip_root_context:
            _ctx["root_ctx"] = self.opres_ctx(dep_on)

        for idx, item in enumerate(plan.task_requests):
            item: TaskRequest
            if idx >= max_agents:
                logging.warning(
                    f"Maximum number of agents ({max_agents}) reached, skipping further requests."
                )
                break

            if (
                auto_context
                and idx != 0
                and plan.execution_strategy == "sequential"
                and set(dep_on) != root_set
            ):
                await self.run_flow(visualize=False)
                _ctx["step_ctx"][f"{idx + 1}"] = self.opres_ctx(dep_on)

            imodel = await self.create_agent_imodel(
                role=item.role, domain=item.domain, **kw
            )
            b_id = self.session.new_branch(
                system_datetime=True,
                use_lion_system_message=True,
                chat_model=imodel,
                parse_model=imodel,
            )
            c_ = plan.common_background + str(item.instruct.context or "")
            c_ = c_.strip()
            ctx = {
                "task_request": item.instruct.instruction,
                "task_context": c_,
                "task_guidance": item.instruct.guidance,
                **_ctx,
            }
            node = self.builder.add_operation(
                "communicate",
                depends_on=dep_on,
                branch=b_id,
                instruction="Perform task, make sure you report all your work in the result message",
                context=ctx,
            )
            nodes.append(node)
            if plan.execution_strategy == "sequential":
                dep_on = [node]

        return nodes

    async def fanout(
        self,
        initial_desc: str,
        planning_instruction: str,
        synth_instruction: str,
        context: str = None,
        visualize: bool | Literal["step", "final"] = False,
        max_agents: int = 8,
    ):
        visualize_step = (
            visualize if isinstance(visualize, bool) else visualize == "step"
        )
        FlowPlansField = self.generate_flow_plans_field(
            initial=initial_desc,
        )
        b = self.new_orc_branch()
        params = {
            "operation": "operate",
            "branch": b.id,
            "field_models": [FlowPlansField],
            "instruct": Instruct(
                reason=True,
                instruction=planning_instruction,
                context=context,
            ),
        }
        if (ln := self.builder.last_operation_id) is not None:
            params["depends_on"] = [ln]

        # 1. establish root node ---------------------------------------------------------
        root = self.builder.add_operation(**params)

        # 2. run planning ----------------------------------------------------------------
        results = await self.run_flow(visualize_step)
        plans = results["operation_results"][root].flow_plans

        # 3. run initial phase ------------------------------------------------------------
        initial_nodes = await self.expand_with_plan(
            root=root,
            plan=plans.initial,
            max_agents=max_agents,
            auto_context=True,
            skip_root_context=True,
        )
        await self.run_flow(visualize_step)

        # 4. synthesis --------------------------------------------------------------------
        imodel = await self.create_orc_imodel()
        b = self.session.new_branch(
            system_datetime=True,
            use_lion_system_message=True,
            system=self.orc_branch.system.clone(),
            chat_model=imodel,
        )

        synth_node = self.builder.add_operation(
            "communicate",
            branch=b.id,
            depends_on=initial_nodes,
            instruction=synth_instruction,
            context=self.opres_ctx(initial_nodes),
        )

        result = await self.run_flow(bool(visualize))
        synth_result = result["operation_results"][synth_node]

        return FanoutResponse(
            synth_node=synth_node,
            synth_result=synth_result,
            flow_results=result,
            initial_nodes=initial_nodes,
        )

    def opres_ctx(self, ops: IDType | list[IDType]) -> dict[str, Any]:
        """Get operation result context for a given operation ID.

        Args:
            session: LionAGI session
            operation: Operation object with branch_id

        Returns:
            Tool usage summary dict
        """
        g = self.builder.get_graph()
        ops = [ops] if not isinstance(ops, list) else ops

        def _get_ctx(op_id):
            try:
                op: Operation = g.internal_nodes[op_id]
                if not op.branch_id:
                    return {
                        "error": f"Operation {str(op_id)} has no branch_id"
                    }

                branch = self.session.get_branch(op.branch_id, None)

                if branch and len(branch.messages) > 0:
                    for i in reversed(list(branch.messages.progression)):
                        if isinstance(
                            msg := branch.messages[i], AssistantResponse
                        ):
                            return {
                                "branch_id": str(branch.id),
                                "branch_name": branch.name,
                                "result": msg.model_response.get(
                                    "result", "N/A"
                                ),
                                "summary": msg.model_response.get(
                                    "summary", "N/A"
                                ),
                            }

                return {"error": "No branch or messages found"}
            except Exception as e:
                return {"error": f"Failed to extract summary: {str(e)}"}

        return [_get_ctx(o) for o in ops]

    async def fanout_w_gated_refinement(
        self,
        initial_desc: str,
        refinement_desc: str,
        gate_instruction: str,
        synth_instruction: str,
        planning_instruction: str,
        context: str = None,
        critic_domain: str = "software-architecture",
        critic_role: str = "critic",
        visualize: bool | Literal["step", "final"] = False,
        gates: list[GateOptions] | dict[str, str] = None,
        max_agents=8,
    ) -> dict:
        """
        Reusable conditional quality-gated workflow pattern.

        Pattern: Planning → Initial Phase → Quality Gate → [Conditional Refinement] → Synthesis

        Args:
            initial_desc: Description for initial phase
            refinement_desc: Description for refinement phase if quality insufficient
            gate_instruction: Instruction for quality gate evaluation
            gate_model: Pydantic model class for quality gate results
            synth_instruction: Instruction for final synthesis
            planning_instruction: Instruction for orchestrator planning
            context: Additional context for planning
            critic_domain: Domain expertise for quality critic
            critic_role: Role for quality critic (default: "critic")
            visualize: Whether to visualize the flow
            gates: List of gate components or dict mapping gate names to descriptions

        Returns:
            dict: Results including final gate, QA branch, refinement execution status,
                synthesis node, synthesis result and flow results.
        """

        # 1. validate inputs ---------------------------------------------------------------
        gate_components = {}
        if gates:
            if isinstance(gates, list):
                gates = {gate: True for gate in gates}
            for g, d in gates.items():
                if isinstance(d, str):
                    gate_components[g] = d
                elif d is True:
                    gate_components[g] = True
                else:
                    raise ValueError(
                        f"Invalid gate component '{g}': expected str or True, got {type(d)}"
                    )
        qa_field = self.generate_quality_gate_field(**gate_components)
        visualize_step = (
            visualize if isinstance(visualize, bool) else visualize == "step"
        )

        # 2. establish root -----------------------------------------------------------------
        plan_field = self.generate_flow_plans_field(
            initial=initial_desc, refinement=refinement_desc
        )
        b = self.new_orc_branch()
        params = {
            "operation": "operate",
            "branch": b.id,
            "field_models": [plan_field],
            "instruct": Instruct(
                reason=True,
                instruction=planning_instruction,
                context=context,
            ),
        }

        if (ln := self.builder.last_operation_id) is not None:
            params["depends_on"] = [ln]

        root_node = self.builder.add_operation(**params)
        results = await self.run_flow(visualize_step)
        plans = results["operation_results"][root_node].flow_plans

        # 3. run initial phase ------------------------------------------------------------
        initial_nodes = await self.expand_with_plan(
            root=root_node,
            plan=plans.initial,
            auto_context=True,
            skip_root_context=True,
            max_agents=max_agents,
        )
        await self.run_flow(visualize_step)

        # 4. run quality gate -

        imodel = await self.create_agent_imodel(
            role=item.role, domain=item.domain
        )

        # ------------------------------------------------------------
        qa_branch = await self.create_cc_branch(
            compose_request=ComposerRequest(
                role=critic_role, domains=critic_domain
            ),
            agent_suffix="_quality_assurant",
            auto_finish=True,
        )
        gate1 = self.builder.add_operation(
            operation="operate",
            branch=qa_branch,
            depends_on=initial_nodes,
            instruct=Instruct(
                instruction=gate_instruction,
                reason=True,
                context=self.opres_ctx(initial_nodes),
            ),
            field_models=[qa_field],
        )
        result = await self.run_flow(visualize_step)

        # 5. evaluate quality gate -------------------------------------------------------
        gate_eval = result["operation_results"][gate1].quality_gate

        final_gate = gate1
        refinement_executed = False
        gate_passed = True

        # 6. conditional refinement --------------------------------------------------------
        if not gate_eval.threshold_met:
            gate_passed = False
            # expand with refinement plan
            refinement_nodes = await self.expand_with_plan(
                root=gate1,
                plan=plans.refinement,
                auto_context=True,
                skip_root_context=False,
            )

            # Second quality gate
            gate2 = self.builder.add_operation(
                "operate",
                branch=qa_branch,
                depends_on=(
                    refinement_nodes
                    if plans.refinement.execution_strategy == "concurrent"
                    else [refinement_nodes[-1]]
                ),
                instruct=Instruct(
                    instruction=gate_instruction,
                    reason=True,
                ),
                field_models=[qa_field],
            )

            res = await self.run_flow(visualize_step)
            if res["operation_results"][gate2].quality_gate.threshold_met:
                gate_passed = True

            final_gate = gate2
            refinement_executed = True

        # 7. synthesis --------------------------------------------------------------------
        b = self.new_orc_branch()
        synth_node = self.builder.add_operation(
            "communicate",
            branch=b.id,
            depends_on=[final_gate],
            instruction=synth_instruction,
        )

        result = await self.run_flow(bool(visualize))
        synth_result = result["operation_results"][synth_node]

        return FanoutWithGatedRefinementResponse(
            synth_node=synth_node,
            synth_result=synth_result,
            flow_results=result,
            initial_nodes=initial_nodes,
            final_gate=final_gate,
            gate_passed=gate_passed,
            refinement_executed=bool(refinement_executed),
        )

    async def save_json(self):
        """Save the current session state to a file."""
        from lionagi.utils import create_path

        fp = create_path(
            directory=f"{cc_settings.REPO}/{cc_settings.WORKSPACE}/{self.flow_name}/snapshots",
            filename=f"{self.flow_name}_session.json",
            dir_exist_ok=True,
            file_exist_ok=True,
            timestamp=True,
        )

        session_meta = self.session.model_dump(
            exclude={
                "branches",
                "default_branch",
                "mail_transfer",
                "mail_manager",
            }
        )

        dict_ = {
            "branches": [
                b.to_dict()
                for b in self.session.branches
                if b != self.session.default_branch
            ],
            "session_default_branch": self.session.default_branch.to_dict(),
            "metadata": session_meta,
            "graph": self.builder.get_graph().to_dict(),
        }

        async with aiofiles.open(fp, "w") as f:
            await f.write(json.dumps(dict_, indent=2))
        logger.info(f"💾 Session saved to {fp}")

    @classmethod
    async def load_json(cls, fp: str | Path):
        """Load session state from a JSON file."""
        fp = Path(fp) if not isinstance(fp, Path) else fp
        if not fp.exists():
            raise FileNotFoundError(f"File {fp} does not exist")

        async with aiofiles.open(fp, "r") as f:
            text = await f.read()

        dict_ = json.loads(text)
        branches = [Branch.from_dict(b) for b in dict_["branches"]]
        orc_branch = Branch.from_dict(dict_["session_default_branch"])

        metadata = {"prev_session_meta": dict_["metadata"]}

        session = Session(
            default_branch=orc_branch,
            metadata=metadata,
            name=dict_["metadata"]["name"],
        )
        session.branches.include(branches)

        internal_nodes = Pile.from_dict(dict_["graph"]["internal_nodes"])
        internal_edges = Pile.from_dict(dict_["graph"]["internal_edges"])
        g = Graph(
            internal_nodes=internal_nodes,
            internal_edges=internal_edges,
        )
        builder = Builder(name=session.name)
        builder.graph = g

        self = cls(flow_name=session.name)
        self.session = session
        self.builder = builder
        return self
