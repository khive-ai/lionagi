#!/usr/bin/env python3
"""Final debug to pinpoint the exact issue."""

import asyncio
import sys
from pathlib import Path

# Add lionagi to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lionagi.services.settings import settings
from lionagi.services.imodel import iModel
import anyio


async def test_with_prints():
    """Test with print statements at every step."""
    
    print("🔍 Final Debug Test\n")
    
    api_key = settings.get_api_key('nvidia_nim')
    
    print("1. Creating iModel...")
    model = iModel(
        provider="nvidia_nim",
        model="meta/llama-3.2-1b-instruct",
        api_key=api_key
    )
    print(f"   ✅ Created")
    
    print("\n2. Starting context manager...")
    async with model as m:
        print("   ✅ Inside context manager")
        
        print("\n3. Creating request...")
        messages = [{"role": "user", "content": "Hi"}]
        request_data = {"messages": messages, "max_tokens": 5}
        
        print("\n4. Before invoke...")
        print(f"   Executor running: {m.executor._running}")
        print(f"   Active calls before: {len(m.executor.active_calls)}")
        
        print("\n5. Calling invoke with timeout...")
        try:
            with anyio.fail_after(15):
                print("   About to await model.invoke()...")
                response = await model.invoke(request_data)
                print(f"   ✅ INVOKE RETURNED: {response}")
                return True
        except TimeoutError:
            print("   ❌ TIMEOUT in invoke")
            print(f"   Active calls: {m.executor.active_calls}")
            print(f"   Completed calls: {m.executor.completed_calls}")
            
            # Check the status of calls
            for call_id, call in m.executor.active_calls.items():
                print(f"   Active: {call_id} = {call.status}")
            for call_id, call in m.executor.completed_calls.items():
                print(f"   Completed: {call_id} = {call.status}")
            return False
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("SHOULD NOT GET HERE")


if __name__ == "__main__":
    try:
        result = asyncio.run(test_with_prints())
        if result:
            print("\n🎉 SUCCESS! API call worked!")
        else:
            print("\n❌ Failed")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()