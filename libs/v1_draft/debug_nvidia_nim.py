#!/usr/bin/env python3
"""Debug script for NVIDIA NIM provider integration."""

import asyncio
import sys
from pathlib import Path

# Add lionagi to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lionagi.services.settings import settings
from lionagi.services.providers.provider_registry import get_provider_registry, register_builtin_adapters
from lionagi.services.imodel import iModel


async def debug_nvidia_nim():
    """Debug NVIDIA NIM provider step by step."""
    
    print("🔍 Debugging NVIDIA NIM Provider\n")
    
    # 1. Test API key
    print("1. Checking API key...")
    api_key = settings.get_api_key("nvidia_nim")
    if not api_key:
        print("   ❌ No API key found")
        return
    print(f"   ✅ API key: {api_key[:8]}...")
    
    # 2. Test registry
    print("\n2. Testing registry...")
    register_builtin_adapters()
    registry = get_provider_registry()
    print(f"   ✅ Adapters: {registry.known_adapters()}")
    
    # 3. Test resolution
    print("\n3. Testing resolution...")
    try:
        resolution, adapter = registry.resolve(
            provider="nvidia_nim", 
            model="meta/llama-3.2-1b-instruct",
            base_url=None
        )
        print(f"   ✅ Resolved: {resolution.adapter_name}")
    except Exception as e:
        print(f"   ❌ Resolution failed: {e}")
        return
    
    # 4. Test service creation
    print("\n4. Testing service creation...")
    try:
        service, res, rights = registry.create_service(
            provider="nvidia_nim",
            model="meta/llama-3.2-1b-instruct",
            base_url=None,
            api_key=api_key
        )
        print(f"   ✅ Service: {service.name}")
    except Exception as e:
        print(f"   ❌ Service creation failed: {e}")
        return
    
    # 5. Test iModel creation (no context manager yet)
    print("\n5. Testing iModel creation...")
    try:
        model = iModel(
            provider="nvidia_nim",
            model="meta/llama-3.2-1b-instruct",
            api_key=api_key
        )
        print(f"   ✅ iModel created: {model.id}")
    except Exception as e:
        print(f"   ❌ iModel creation failed: {e}")
        return
    
    # 6. Test context manager entry
    print("\n6. Testing context manager entry...")
    try:
        print("   🚀 Entering context manager...")
        await model.__aenter__()
        print("   ✅ Context manager entered successfully")
        
        # 7. Test simple API call
        print("\n7. Testing API call...")
        try:
            messages = [{"role": "user", "content": "Hi"}]
            request_data = {"messages": messages}
            print("   🚀 Making API call...")
            
            # Set a very short timeout to avoid hanging
            import anyio
            with anyio.fail_after(10):  # 10 second timeout
                response = await model.invoke(request_data)
            
            print(f"   ✅ Response received: {type(response)}")
            if isinstance(response, dict) and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                print(f"   📝 Content: {content[:50]}...")
            
        except Exception as e:
            print(f"   ❌ API call failed: {e}")
        
        # 8. Test context manager exit
        print("\n8. Testing context manager exit...")
        try:
            await model.__aexit__(None, None, None)
            print("   ✅ Context manager exited successfully")
        except Exception as e:
            print(f"   ❌ Context manager exit failed: {e}")
            
    except Exception as e:
        print(f"   ❌ Context manager entry failed: {e}")
        return
    
    print("\n🎉 Debug completed!")


if __name__ == "__main__":
    try:
        asyncio.run(debug_nvidia_nim())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()