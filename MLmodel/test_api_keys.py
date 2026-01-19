"""
Test script to validate API keys locally before deployment
"""
import os
import requests
import json

def test_google_gemini():
    """Test Google Gemini API"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ GOOGLE_API_KEY not set in environment")
        return False
    
    print(f"✓ Testing Google Gemini API (key: ...{api_key[-10:]})")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": "Say 'API key is valid' if you can read this."}]
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 50
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            result = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            print(f"✅ Google Gemini API: WORKING")
            print(f"   Response: {result}")
            return True
        else:
            print(f"❌ Google Gemini API: FAILED (Status {response.status_code})")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Google Gemini API: ERROR - {str(e)}")
        return False


def test_openrouter(key_name, api_key):
    """Test OpenRouter API"""
    if not api_key:
        print(f"❌ {key_name} not set in environment")
        return False
    
    print(f"✓ Testing {key_name} (key: ...{api_key[-10:]})")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "xiaomi/mimo-v2-flash:free",
        "messages": [{"role": "user", "content": "Say 'API key is valid' if you can read this."}],
        "temperature": 0,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ {key_name}: WORKING")
            print(f"   Response: {result}")
            return True
        else:
            print(f"❌ {key_name}: FAILED (Status {response.status_code})")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {key_name}: ERROR - {str(e)}")
        return False


def main():
    print("=" * 60)
    print("API KEY VALIDATION TEST")
    print("=" * 60)
    print()
    
    results = {}
    
    # Test Google Gemini
    print("1. Testing Google Gemini API (for backend column analysis)")
    print("-" * 60)
    results['google'] = test_google_gemini()
    print()
    
    # Test OpenRouter Key 1
    print("2. Testing OpenRouter Key 1 (for frontend LLM)")
    print("-" * 60)
    openrouter_key1 = os.getenv("NEXT_PUBLIC_OPENROUTER_KEY_1") or os.getenv("OPENROUTER_KEY_1")
    results['openrouter1'] = test_openrouter("OPENROUTER_KEY_1", openrouter_key1)
    print()
    
    # Test OpenRouter Key 2
    print("3. Testing OpenRouter Key 2 (for frontend LLM)")
    print("-" * 60)
    openrouter_key2 = os.getenv("NEXT_PUBLIC_OPENROUTER_KEY_2") or os.getenv("OPENROUTER_KEY_2")
    results['openrouter2'] = test_openrouter("OPENROUTER_KEY_2", openrouter_key2)
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(results.values())
    
    for key, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {key}: {'PASS' if status else 'FAIL'}")
    
    print()
    print(f"Results: {passed}/{total} API keys working")
    
    if passed == total:
        print("\n🎉 All API keys are valid! Ready to deploy.")
    else:
        print("\n⚠️  Some API keys failed. Please fix before deploying.")
        print("\nTo set environment variables:")
        print("  Windows PowerShell:")
        print("    $env:GOOGLE_API_KEY='your-key-here'")
        print("    $env:OPENROUTER_KEY_1='your-key-here'")
        print("    $env:OPENROUTER_KEY_2='your-key-here'")
        print("\n  Or create a .env file in MLmodel/ with:")
        print("    GOOGLE_API_KEY=your-key-here")
        print("    OPENROUTER_KEY_1=your-key-here")
        print("    OPENROUTER_KEY_2=your-key-here")


if __name__ == "__main__":
    main()
