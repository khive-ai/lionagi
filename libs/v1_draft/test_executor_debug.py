#!/usr/bin/env python3
"""Debug exactly where the executor is hanging."""

import asyncio
import sys
from pathlib import Path

# Add lionagi to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lionagi.services.settings import settings
from lionagi.services.imodel import iModel
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


async def test_executor_detailed():
    """Test executor with detailed logging."""
    
    print("🔍 Detailed Executor Debug\n")
    
    api_key = settings.get_api_key('nvidia_nim')
    if not api_key:
        print("❌ No API key")
        return
    
    print("1. Creating iModel...")
    model = iModel(
        provider="nvidia_nim",
        model="meta/llama-3.2-1b-instruct",
        api_key=api_key
    )
    print(f"   ✅ Created: {model.id}")
    
    print("\n2. Entering context manager...")
    async with model as m:
        print(f"   ✅ Context entered")
        print(f"   ✅ Executor running: {m.executor._running}")
        
        print("\n3. Creating request...")
        messages = [{"role": "user", "content": "Hi"}]
        request_data = {"messages": messages, "max_tokens": 5}
        
        print("\n4. Calling invoke (this is where it hangs)...")
        print("   🚀 About to call model.invoke()...")
        
        import anyio
        try:
            with anyio.fail_after(10):
                print("   ⏰ Starting with 10 second timeout...")
                response = await model.invoke(request_data)
                print(f"   ✅ Got response: {response}")
        except TimeoutError:
            print("   ❌ Timed out in invoke!")
            print(f"   📊 Active calls: {len(model.executor.active_calls)}")
            print(f"   📊 Completed calls: {len(model.executor.completed_calls)}")
            print(f"   📊 Queue size: {model.executor._queue.size()}")
            
            # Check if there are any calls stuck
            for call_id, call in model.executor.active_calls.items():
                print(f"   🔍 Active call {call_id}: status={call.status}")
                
            # Force exit
            return
    
    print("\n🎉 Success!")


if __name__ == "__main__":
    try:
        asyncio.run(test_executor_detailed())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()