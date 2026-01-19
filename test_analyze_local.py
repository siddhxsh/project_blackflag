"""
Test /analyze endpoint by directly calling the analysis without file upload
"""
import requests

url = "http://127.0.0.1:5000/analyze"

# Upload the amazon.csv file
with open(r"C:\Users\Admin\OneDrive\Desktop\e-com\MLmodel\data\amazon.csv", 'rb') as f:
    files = {'file': ('amazon.csv', f, 'text/csv')}
    
    print("Uploading amazon.csv...")
    print("This may take 1-2 minutes for LLM column analysis...")
    
    response = requests.post(url, files=files, timeout=180)
    
    if response.status_code == 200:
        result = response.json()
        print("\n" + "="*60)
        print("✓ SUCCESS!")
        print("="*60)
        print(f"Status: {result['status']}")
        print(f"Total Reviews: {result['summary']['total_reviews']}")
        print(f"\nSentiment Summary:")
        print(f"  Positive: {result['summary']['sentiment_summary']['positive']}")
        print(f"  Negative: {result['summary']['sentiment_summary']['negative']}")
        print(f"  Neutral: {result['summary']['sentiment_summary']['neutral']}")
        print(f"\nOutput Files:")
        for key, filename in result['output_files'].items():
            print(f"  - {filename}")
        print("="*60)
    else:
        print(f"\n✗ ERROR {response.status_code}")
        print(response.json())
