from lionagi import Branch, iModel, load_mcp_tools


async def main():

    tools = await load_mcp_tools(
        "/Users/lion/projects/lionagi/cookbooks/using_mcp/.mcp.json"
    )

    gpt = iModel(provider="openai", model="gpt-5-mini")
    branch = Branch(chat_model=gpt, tools=tools)
    print(f"tools: {list(branch.tools.keys())}")

    try:
        await branch.ReAct(
            instruct={
                "instruction": (
                    "Research the latest developments in Model Context Protocol (MCP). "
                    "Use the search tools to find recent information about MCP, "
                    "its features, and adoption by different platforms."
                ),
                "guidance": """tool use example:  
                - {'function': 'search', 'arguments': {'request': 'PerplexityChatRequest(\n  model=\"sonar\",\n  messages=[{\"role\": \"user\", \"content\": \"What are the latest developments in MCP (Model Context Protocol) from Anthropic in 2025?\"}], return_related_questions=True)'}}
                       
                - {'function': 'search', 'arguments': {'request': ExaSearchRequest(\n  query=\"agentic AI frameworks Python 2025\",\n  type=\"neural\",\n numResults=5, useAutoprompt=True)"}}
                """,
            },
            max_extensions=3,
            verbose=True,
        )

    except Exception as e:
        print(f"\n❌ ReAct failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import anyio

    anyio.run(main)
