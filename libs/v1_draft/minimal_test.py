#!/usr/bin/env python3
"""Minimal test to isolate the hanging issue."""

import asyncio
import sys
from pathlib import Path

# Add lionagi to path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lionagi.services.settings import settings
from lionagi.services.imodel import iModel


async def minimal_test():
    """Minimal test of async context manager."""
    
    print("🔍 Minimal async test")
    
    api_key = settings.get_api_key('nvidia_nim')
    if not api_key:
        print("❌ No API key")
        return
    
    print("✅ Creating iModel...")
    model = iModel(
        provider="nvidia_nim",
        model="meta/llama-3.2-1b-instruct",
        api_key=api_key
    )
    print(f"✅ Created: {model.id}")
    
    print("🚀 Testing context manager...")
    try:
        print("   Entering...")
        async with model as m:
            print(f"   ✅ Entered successfully: {m.id}")
            print("   ✅ Context manager working!")
        print("   ✅ Exited successfully")
    except Exception as e:
        print(f"   ❌ Context manager failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎉 Minimal test completed!")


if __name__ == "__main__":
    try:
        asyncio.run(minimal_test())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()