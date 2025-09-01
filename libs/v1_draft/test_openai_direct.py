#!/usr/bin/env python3
"""Test OpenAI client directly against NVIDIA NIM to isolate the issue."""

import asyncio
import sys
from pathlib import Path

# Add lionagi to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lionagi.services.settings import settings


async def test_openai_direct():
    """Test OpenAI client directly against NVIDIA NIM."""
    
    print("🔍 Testing OpenAI Client Direct to NVIDIA NIM\n")
    
    # 1. Get API key
    api_key = settings.get_api_key('nvidia_nim')
    if not api_key:
        print("❌ No API key found")
        return False
    print(f"✅ API key: {api_key[:8]}...")
    
    # 2. Test with raw OpenAI client
    print("\n2. Testing raw OpenAI AsyncClient...")
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            timeout=10.0  # 10 second timeout
        )
        
        print("   ✅ Client created")
        
        # 3. Make a simple API call
        print("   🚀 Making API call...")
        
        # Add timeout at asyncio level too
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="meta/llama-3.2-1b-instruct",
                messages=[
                    {"role": "user", "content": "Hello! Just say 'Hi' back."}
                ],
                max_tokens=10
            ),
            timeout=15.0
        )
        
        print(f"   ✅ API call successful!")
        print(f"   📝 Response: {response.choices[0].message.content}")
        print(f"   🤖 Model: {response.model}")
        
        return True
        
    except asyncio.TimeoutError:
        print("   ❌ API call timed out - likely network/server issue")
        return False
    except Exception as e:
        print(f"   ❌ API call failed: {e}")
        print(f"   🔍 Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_httpx_direct():
    """Test HTTPX directly against NVIDIA NIM."""
    
    print("\n3. Testing raw HTTPX client...")
    try:
        import httpx
        
        api_key = settings.get_api_key('nvidia_nim')
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            print("   ✅ HTTPX client created")
            
            response = await client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta/llama-3.2-1b-instruct",
                    "messages": [
                        {"role": "user", "content": "Hello! Just say 'Hi' back."}
                    ],
                    "max_tokens": 10
                }
            )
            
            print(f"   ✅ HTTPX call successful! Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                print(f"   📝 Response: {content}")
                return True
            else:
                print(f"   ❌ HTTP error: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"   ❌ HTTPX call failed: {e}")
        return False


async def main():
    """Run all tests."""
    try:
        openai_success = await test_openai_direct()
        httpx_success = await test_httpx_direct()
        
        if openai_success and httpx_success:
            print("\n🎉 Both OpenAI and HTTPX work - issue is in our wrapper!")
        elif httpx_success and not openai_success:
            print("\n🔍 HTTPX works but OpenAI SDK hangs - OpenAI SDK issue")
        elif not httpx_success:
            print("\n🔍 Both fail - likely network/NVIDIA NIM endpoint issue")
        else:
            print("\n🤔 Mixed results - need more investigation")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())