#!/usr/bin/env python3
"""Trace the exact issue by adding detailed logging."""

import asyncio
import sys
from pathlib import Path

# Add lionagi to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lionagi.services.settings import settings
from lionagi.services.imodel import iModel
import logging

# Enable VERY detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)


async def trace_issue():
    """Trace exactly where the flow breaks."""
    
    print("\n🔍 TRACING THE ISSUE\n")
    
    api_key = settings.get_api_key('nvidia_nim')
    
    print("1. Creating iModel...")
    model = iModel(
        provider="nvidia_nim",
        model="meta/llama-3.2-1b-instruct",
        api_key=api_key
    )
    
    print("\n2. Entering context...")
    async with model as m:
        print("   ✅ Context entered")
        
        print("\n3. Before invoke...")
        messages = [{"role": "user", "content": "Hi"}]
        request_data = {"messages": messages, "max_tokens": 5}
        
        print("\n4. Calling invoke...")
        print("   📍 About to call executor.submit_call()")
        
        # Let's manually trace what invoke does
        from lionagi.services.endpoint import ChatRequestModel
        request = ChatRequestModel(**request_data)
        print(f"   📍 Request created: {request.model}")
        
        # Build context
        from uuid import uuid4
        context = m._build_context()
        print(f"   📍 Context created: {context.call_id}")
        
        # Submit to executor
        print("   📍 Submitting to executor...")
        call = await m.executor.submit_call(m.service, request, context)
        print(f"   📍 Call submitted: {call.id}")
        print(f"   📍 Call status: {call.status}")
        print(f"   📍 Completion event exists: {call._completion_event is not None}")
        
        # Wait for completion
        print("\n5. Waiting for completion...")
        print(f"   📍 Before wait_completion()")
        
        import anyio
        try:
            with anyio.fail_after(5):
                print("   📍 Starting wait with 5s timeout...")
                result = await call.wait_completion()
                print(f"   ✅ GOT RESULT: {result}")
                return True
        except TimeoutError:
            print("   ❌ TIMEOUT in wait_completion()")
            print(f"   📍 Call status: {call.status}")
            print(f"   📍 Call has result: {hasattr(call, 'result')}")
            if hasattr(call, 'result'):
                print(f"   📍 Call result: {call.result}")
            print(f"   📍 Event is set: {call._completion_event.is_set()}")
            
            # Check executor state
            print(f"\n   📍 Active calls: {len(m.executor.active_calls)}")
            print(f"   📍 Completed calls: {len(m.executor.completed_calls)}")
            
            for cid, c in m.executor.active_calls.items():
                print(f"      Active: {cid} = {c.status}")
            for cid, c in m.executor.completed_calls.items():
                print(f"      Completed: {cid} = {c.status}")
                
            return False
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    result = asyncio.run(trace_issue())
    if result:
        print("\n🎉 SUCCESS!")
    else:
        print("\n❌ Failed - see trace above")