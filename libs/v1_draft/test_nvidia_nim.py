#!/usr/bin/env python3
"""Test script for NVIDIA NIM provider integration."""

import asyncio
import os
import sys
from pathlib import Path

# Add lionagi to path
sys.path.insert(0, str(Path(__file__).parent / "libs" / "lionagi" / "src"))

from lionagi.services.settings import settings
from lionagi.services.providers.provider_registry import get_provider_registry, register_builtin_adapters
from lionagi.services.imodel import iModel


async def test_nvidia_nim_integration():
    """Test NVIDIA NIM provider integration with actual API call."""
    
    print("🧪 Testing NVIDIA NIM Provider Integration\n")
    
    # 1. Test settings detection
    print("1. Testing API key detection...")
    api_key = settings.get_api_key("nvidia_nim")
    if api_key:
        print(f"   ✅ NVIDIA NIM API key detected: {api_key[:8]}...")
    else:
        print("   ❌ No NVIDIA NIM API key found")
        print("   💡 Set LIONAGI_NVIDIA_NIM_API_KEY environment variable")
        return False
    
    print()
    
    # 2. Test adapter registration
    print("2. Testing adapter registration...")
    register_builtin_adapters()
    registry = get_provider_registry()
    
    if "nvidia_nim" in registry.known_adapters():
        print("   ✅ NVIDIA NIM adapter registered")
    else:
        print("   ❌ NVIDIA NIM adapter not found")
        return False
    
    print()
    
    # 3. Test provider resolution
    print("3. Testing provider resolution...")
    try:
        resolution, adapter = registry.resolve(
            provider="nvidia_nim", 
            model="meta/llama-3.2-1b-instruct",
            base_url=None
        )
        print(f"   ✅ Provider resolved: {resolution.provider}")
        print(f"   ✅ Adapter name: {resolution.adapter_name}")
        print(f"   ✅ Base URL: {resolution.base_url}")
    except Exception as e:
        print(f"   ❌ Resolution failed: {e}")
        return False
        
    print()
    
    # 4. Test service creation
    print("4. Testing service creation...")
    try:
        service, res, rights = registry.create_service(
            provider="nvidia_nim",
            model="meta/llama-3.2-1b-instruct",
            base_url=None,
            api_key=api_key
        )
        print(f"   ✅ Service created: {service.name}")
        print(f"   ✅ Required rights: {rights}")
    except Exception as e:
        print(f"   ❌ Service creation failed: {e}")
        return False
        
    print()
    
    # 5. Test iModel creation and API call
    print("5. Testing actual API call...")
    try:
        # Create iModel with NVIDIA NIM provider
        async with iModel(
            provider="nvidia_nim",
            model="meta/llama-3.2-1b-instruct",
            api_key=api_key
        ) as model:
            # Make a simple completion call
            messages = [
                {"role": "user", "content": "Hello! Please respond with just 'Hi there!' to test the connection."}
            ]
            
            print("   🚀 Making API call...")
            request_data = {"messages": messages}
            response = await model.invoke(request_data)
            
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                print(f"   ✅ API call successful!")
                print(f"   📝 Response: {content}")
                
                # Check model info
                model_used = response.get("model", "unknown")
                print(f"   🤖 Model used: {model_used}")
                
            else:
                print(f"   ⚠️  Unexpected response format: {response}")
                
    except Exception as e:
        print(f"   ❌ API call failed: {e}")
        return False
        
    print()
    
    # 6. Test alternative provider names
    print("6. Testing alternative provider names...")
    try:
        # Test "nvidia" as provider name
        async with iModel(
            provider="nvidia", 
            model="meta/llama-3.2-1b-instruct",
            api_key=api_key
        ) as model_nvidia:
            print("   ✅ 'nvidia' provider name works")
        
        # Test model prefix
        async with iModel(
            model="nvidia/meta/llama-3.2-1b-instruct",
            api_key=api_key
        ) as model_prefix:
            print("   ✅ 'nvidia/' model prefix works")
        
    except Exception as e:
        print(f"   ❌ Alternative names failed: {e}")
        return False
        
    print()
    print("🎉 All tests passed! NVIDIA NIM provider integration is working correctly.")
    return True


async def main():
    """Run the test."""
    try:
        success = await test_nvidia_nim_integration()
        if success:
            print("\n✅ NVIDIA NIM provider is ready to use!")
            print("\nUsage examples:")
            print("  model = iModel(provider='nvidia_nim', model='meta/llama-3.1-8b-instruct')")
            print("  model = iModel(provider='nvidia', model='meta/llama-3.1-70b-instruct')")
            print("  model = iModel(model='nvidia/mistralai/mistral-7b-instruct-v0.3')")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Check the output above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())