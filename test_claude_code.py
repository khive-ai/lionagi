from lionagi import iModel, Branch
from lionagi.service.providers.anthropic_.claude_code.claude_code import ClaudeCodeEndpoint
from lionagi.service.types import APICalling

async def main():
    # The correct way: pass the messages and other params directly as kwargs
    try:
        # First, let's test the endpoint directly
        print("Testing api calling directly...")
        request_dict = {
            "messages": [
                {"role": "user",
                "content": "Hello, Claude Code! Can you give me a high level overview of lionagi?"}
            ],
            "allowed_tools": ["Write", "Read", "Edit"],
            "model": "claude-sonnet-4-20250514",
        }
        
        # endpoint = ClaudeCodeEndpoint()
        # response = await endpoint.call(request_dict)
        
        
        # # api_call = APICalling(endpoint=endpoint, payload=request_dict)
        # # await api_call.invoke()
        # # response = api_call.response
        
        imodel = iModel(
            provider="claude_code",
            endpoint="code",
            model="claude-sonnet-4-20250514",
            api_key="dummy_api_key",
            allowed_tools=["Write", "Read", "Edit"],
            max_turns=1,
        )
        
        branch = Branch(chat_model=imodel)
        response = await branch.chat("Hello, Claude Code! Can you give me a high level overview of lionagi?")
        
        # response = await imodel.invoke(**request_dict)
        # response = response.response

        # print("Parsed response:", response)
        # print()
        
        # # Now test through iModel
        # print("Testing through iModel...")
        # response = await imodel.invoke(
        #     messages=[
        #         {"role": "user",
        #         "content": "Hello, Claude Code! Can you help me with a Python script?"}
        #     ],
        #     allowed_tools=["Write", "Read", "Edit"]
        # )
        
        print("Response:", response)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # # Extract session ID if available
    # if isinstance(response, dict) and "session_id" in response:
    #     print(f"Session ID: {response['session_id']}")
        
    #     # Test resuming the session
    #     print("\nResuming session...")
    #     response2 = await imodel.invoke(
    #         messages=[
    #             {"role": "user", 
    #             "content": "What did I just ask you about?"}
    #         ],
    #         session_id=response['session_id']
    #     )
    #     print("Resume response:", response2)
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
