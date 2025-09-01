#!/usr/bin/env python3
"""Test to isolate where in the service layer the hanging occurs."""

import asyncio
import sys
from pathlib import Path

# Add lionagi to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lionagi.services.settings import settings
from lionagi.services.providers.provider_registry import get_provider_registry, register_builtin_adapters
from lionagi.services.endpoint import ChatRequestModel
from lionagi.services.core import CallContext
from lionagi.services.imodel import iModel
import anyio


async def test_service_layer():
    """Test each layer of the service stack to find where it hangs."""
    
    print("🔍 Testing Service Layer Components\n")
    
    api_key = settings.get_api_key('nvidia_nim')
    
    # 1. Test service creation
    print("1. Testing service creation...")
    register_builtin_adapters()
    registry = get_provider_registry()
    
    service, res, rights = registry.create_service(
        provider="nvidia_nim",
        model="meta/llama-3.2-1b-instruct",
        base_url=None,
        api_key=api_key
    )
    print(f"   ✅ Service created: {service.name}")
    
    # 2. Test request model creation
    print("\n2. Testing request model...")
    request = ChatRequestModel(
        model="meta/llama-3.2-1b-instruct",
        messages=[{"role": "user", "content": "Hello! Just say 'Hi' back."}],
        max_tokens=10
    )
    print(f"   ✅ Request model: {request.model}")
    
    # 3. Test call context creation
    print("\n3. Testing call context...")
    from uuid import uuid4
    context = CallContext.new(
        branch_id=uuid4(),
        deadline_s=None,
        capabilities=frozenset({"net.out:integrate.api.nvidia.com"})
    )
    print(f"   ✅ Context: {context.call_id}")
    
    # 4. Test direct service call (bypassing executor)
    print("\n4. Testing direct service call...")
    try:
        print("   🚀 Making direct service call...")
        with anyio.fail_after(10):  # 10 second timeout
            response = await service.call(request, ctx=context)
        print(f"   ✅ Direct service call successful!")
        if "choices" in response:
            content = response["choices"][0]["message"]["content"]
            print(f"   📝 Response: {content}")
        return True
    except Exception as e:
        print(f"   ❌ Direct service call failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_executor_layer():
    """Test the executor layer specifically."""
    
    print("\n5. Testing executor layer...")
    try:
        # Create iModel and test executor operations
        api_key = settings.get_api_key('nvidia_nim')
        
        async with iModel(
            provider="nvidia_nim",
            model="meta/llama-3.2-1b-instruct",
            api_key=api_key
        ) as model:
            print("   ✅ iModel context entered")
            
            # Test executor state
            print(f"   ✅ Executor running: {model.executor._running}")
            print(f"   ✅ Active calls: {len(model.executor.active_calls)}")
            
            # Try invoke with timeout
            print("   🚀 Testing invoke with timeout...")
            
            messages = [{"role": "user", "content": "Hello! Just say 'Hi' back."}]
            request_data = {"messages": messages, "max_tokens": 10}
            
            # Use anyio timeout
            with anyio.fail_after(15):
                response = await model.invoke(request_data)
                
            print(f"   ✅ Invoke successful!")
            if isinstance(response, dict) and "choices" in response:
                content = response["choices"][0]["message"]["content"] 
                print(f"   📝 Response: {content}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Executor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    try:
        print("Testing where the service layer hangs...\n")
        
        service_success = await test_service_layer()
        
        if service_success:
            print("\n✅ Direct service calls work!")
            executor_success = await test_executor_layer()
            
            if executor_success:
                print("\n🎉 Everything works! Issue was likely timing/timeout related.")
            else:
                print("\n🔍 Issue is in the executor layer - likely task processing or queuing")
        else:
            print("\n🔍 Issue is in the service layer itself")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user") 
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())