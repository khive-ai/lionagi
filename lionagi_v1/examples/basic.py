import asyncio

from lionagi_v1.base.graph import OpGraph, OpNode
from lionagi_v1.base.ipu import StrictIPU, default_invariants
from lionagi_v1.base.runner import Runner
from lionagi_v1.base.types import Branch, Capability
from lionagi_v1.examples.llm_morphism import LLMCall


class StubProvider:
    async def generate(self, prompt: str) -> str:
        return f"[stub] {prompt}"


async def main():
    br = Branch(name="demo")
    # Capabilities: allow net.out for LLMCall
    br.caps = (Capability(subject=br.id, rights={"net.out"}),)

    n1 = OpNode(m=LLMCall(StubProvider()))
    g = OpGraph(nodes={n1.id: n1}, roots={n1.id})

    runner = Runner(ipu=StrictIPU(default_invariants()))
    results = await runner.run(br, g)
    print("Results:", results)
    print("Branch ctx:", br.ctx)


if __name__ == "__main__":
    asyncio.run(main())
