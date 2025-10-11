from typing import Any, Dict, Set

from lionagi_v1.base.types import Branch


class LLMCall:
    name = "llm.call"
    requires: Set[str] = {"net.out"}  # declare rights

    def __init__(self, provider):
        self._prov = provider  # provider.generate(prompt) -> str

    async def pre(self, br: Branch, **kw) -> bool:
        return "prompt" in kw and isinstance(kw["prompt"], str)

    async def apply(self, br: Branch, **kw) -> Dict[str, Any]:
        text = await self._prov.generate(kw["prompt"])
        br.ctx["last_llm"] = text
        return {"text": text}

    async def post(self, br: Branch, result: Dict[str, Any]) -> bool:
        return "text" in result and isinstance(result["text"], str)
