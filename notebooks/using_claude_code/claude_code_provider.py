from lionagi import iModel, Branch

async def main():
    try:
        imodel = iModel(
            provider="claude_code",
            endpoint="code",
            model="claude-sonnet-4-20250514",
            api_key="dummy_api_key",
            allowed_tools=["Write", "Read", "Edit"],
        )
        
        branch = Branch(chat_model=imodel)
        ins, res = await branch.chat(
            "Hello, Claude Code! Can you give me a brief high level overview of lionagi based on what you can find? Keep it concise in one paragraph.",
            return_ins_res_message=True,
        )
        
        print("Instruction:", ins)
        print("Response object:", res)
        print("Response content:", res.response if res else None)
        
        # Test auto-resume functionality
        print("\n\nTesting auto-resume with same iModel...")
        ins2, res2 = await branch.chat(
            "What did I just ask you about?",
            return_ins_res_message=True,
        )
        print("Second response:", res2.response if res2 else None)
        
        # Check if session was resumed
        print("\nStored session_id:", imodel._provider_metadata.get("session_id"))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())