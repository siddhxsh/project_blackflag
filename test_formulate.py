import requests
import json

# API endpoint
url = "http://127.0.0.1:5000/formulate"

# Test with different LLM providers
providers = [
    {"llm_provider": "openrouter", "model": "xiaomi/mimo-v2-flash:free"}
]

for provider_config in providers:
    print(f"\n{'='*60}")
    print(f"Testing with provider: {provider_config.get('llm_provider')}")
    print(f"{'='*60}\n")
    
    try:
        response = requests.post(
            url,
            json=provider_config,
            timeout=120  # 2 minutes for LLM processing
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ SUCCESS!\n")
            print(f"LLM Provider: {result['llm_provider']}")
            print(f"Total Reviews: {result['total_reviews']:,}")
            print(f"\nSentiment Summary:")
            for sentiment, count in result['sentiment_summary'].items():
                print(f"  {sentiment}: {count:,}")
            
            print(f"\n{'='*60}")
            print("EXECUTIVE SUMMARY:")
            print(f"{'='*60}")
            print(result['executive_summary'])
            print(f"\n{'='*60}")
        else:
            print(f"✗ ERROR {response.status_code}")
            print(response.json())
    
    except requests.exceptions.Timeout:
        print("✗ Request timed out (exceeded 120 seconds)")
    except Exception as e:
        print(f"✗ Error: {e}")
