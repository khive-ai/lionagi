#!/usr/bin/env python3
"""Simple debug to find the exact hanging point."""

import asyncio
import sys
from pathlib import Path

# Add lionagi to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_each_step():
    """Test each step individually."""
    
    print("🔍 Step-by-step debug\n")
    
    try:
        # Step 1: Settings
        print("1. Testing settings...")
        from lionagi.services.settings import settings
        api_key = settings.get_api_key('nvidia_nim')
        print(f"   ✅ API key: {api_key[:8]}...")
        
        # Step 2: Registry  
        print("\n2. Testing registry...")
        from lionagi.services.providers.provider_registry import get_provider_registry, register_builtin_adapters
        register_builtin_adapters()
        registry = get_provider_registry()
        print("   ✅ Registry ready")
        
        # Step 3: Service creation
        print("\n3. Testing service creation...")
        service, res, rights = registry.create_service(
            provider="nvidia_nim",
            model="meta/llama-3.2-1b-instruct", 
            base_url=None,
            api_key=api_key
        )
        print(f"   ✅ Service: {service.name}")
        
        # Step 4: Direct OpenAI client test within service
        print("\n4. Testing service's OpenAI client...")
        print(f"   ✅ Service client: {type(service.client)}")
        print(f"   ✅ Base URL: {service.client.base_url}")
        
        # Step 5: Create request model
        print("\n5. Testing request model...")
        from lionagi.services.endpoint import ChatRequestModel
        request = ChatRequestModel(
            model="meta/llama-3.2-1b-instruct",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        print("   ✅ Request model created")
        
        # Step 6: Create context
        print("\n6. Testing call context...")
        from lionagi.services.core import CallContext
        from uuid import uuid4
        context = CallContext.new(
            branch_id=uuid4(),
            capabilities=frozenset({"net.out:integrate.api.nvidia.com"})
        )
        print("   ✅ Context created")
        
        # Step 7: Test direct service call with timeout
        print("\n7. Testing direct service call...")
        print("   🚀 Making service.call()...")
        
        import anyio
        with anyio.fail_after(10):
            result = await service.call(request, ctx=context)
            
        print("   ✅ Direct service call worked!")
        if isinstance(result, dict) and "choices" in result:
            content = result["choices"][0]["message"]["content"]
            print(f"   📝 Response: {content}")
            
        print("\n🎉 Direct service layer works! Issue is in executor.")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed at current step: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        asyncio.run(test_each_step())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()