#!/usr/bin/env python
import requests
import json

url = 'https://project-blackflag.onrender.com/analyze'
file_path = r'C:\Users\Admin\OneDrive\Desktop\e-com\MLmodel\data\amazon.csv'

print(f"Uploading {file_path} to {url}")
try:
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files, timeout=120)
        print(f'Status Code: {response.status_code}')
        print(f'Response Size: {len(response.content)} bytes')
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nStatus: {data.get('status')}")
            print(f"Message: {data.get('message')}")
            if 'summary' in data:
                print(f"Summary: {json.dumps(data['summary'], indent=2)}")
            if 'output_files' in data:
                print(f"Output Files: {list(data['output_files'].keys())}")
        else:
            print(f"Error Response: {response.text[:500]}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {str(e)}")
