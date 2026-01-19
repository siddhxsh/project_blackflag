import os
import sys
import json
import re
import pandas as pd
import requests


def load_csv_data(csv_path: str) -> tuple[list[str], list]:
    """Load CSV file and return column names and first 5 rows."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    column_names = df.columns.tolist()
    first_rows = df.head(5).values.tolist()
    
    return column_names, first_rows


def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON object from response using regex, with defensive parsing."""
    # Try to find JSON object pattern
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response_text)
    
    if not matches:
        raise ValueError(f"No JSON object found in response: {response_text}")
    
    # Try the longest match first (most likely to be complete)
    for match in sorted(matches, key=len, reverse=True):
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    
    raise ValueError(f"Could not parse any valid JSON from response: {response_text}")


def call_google_gemini_api(column_names: list[str], first_rows: list, api_key: str) -> dict:
    """Call Google Gemini API to map column names to required fields."""
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Prepare the data summary for the prompt
    csv_summary = f"Column names: {column_names}\n\nFirst 5 rows:\n"
    for i, row in enumerate(first_rows):
        csv_summary += f"Row {i+1}: {row}\n"
    
    prompt = f"""Analyze this CSV data and map the column names to the exact required fields.

{csv_summary}

Return ONLY a valid JSON object with these exact keys:
- ProductName
- Price
- Rate
- Review
- Summary

Each value must be a column name from the CSV (case-sensitive) or null if the field doesn't exist.

Example response format (no markdown, no explanation):
{{"ProductName": "product_name", "Price": "price", "Rate": "rating", "Review": "review_text", "Summary": "title"}}

Response:"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json",
    }
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 500
        }
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    
    response.raise_for_status()
    response_data = response.json()
    
    if "candidates" not in response_data or len(response_data["candidates"]) == 0:
        raise ValueError(f"Invalid API response: {response_data}")
    
    content = response_data["candidates"][0]["content"]["parts"][0]["text"]
    return extract_json_from_response(content)


def call_openrouter_api(column_names: list[str], first_rows: list, api_key: str) -> dict:
    """Call OpenRouter API to map column names to required fields."""
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # Prepare the data summary for the prompt
    csv_summary = f"Column names: {column_names}\n\nFirst 5 rows:\n"
    for i, row in enumerate(first_rows):
        csv_summary += f"Row {i+1}: {row}\n"
    
    prompt = f"""Analyze this CSV data and map the column names to the exact required fields.

{csv_summary}

Return ONLY a valid JSON object with these exact keys:
- ProductName
- Price
- Rate
- Review
- Summary

Each value must be a column name from the CSV (case-sensitive) or null if the field doesn't exist.

Example response format (no markdown, no explanation):
{{"ProductName": "product_name", "Price": "price", "Rate": "rating", "Review": "review_text", "Summary": "title"}}

Response:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "xiaomi/mimo-v2-flash:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    
    response.raise_for_status()
    response_data = response.json()
    
    if "choices" not in response_data or len(response_data["choices"]) == 0:
        raise ValueError(f"Invalid API response: {response_data}")
    
    content = response_data["choices"][0]["message"]["content"]
    return extract_json_from_response(content)


def validate_mapping(mapping: dict, column_names: list[str]) -> dict:
    """Validate that all values in mapping are either null or valid column names."""
    required_keys = {"ProductName", "Price", "Rate", "Review", "Summary"}
    
    if set(mapping.keys()) != required_keys:
        raise ValueError(f"Mapping keys do not match required keys. Got: {set(mapping.keys())}, Expected: {required_keys}")
    
    for key, value in mapping.items():
        if value is not None and value not in column_names:
            raise ValueError(f"Column '{value}' mapped to '{key}' not found in CSV columns: {column_names}")
    
    return mapping


def analyze_columns_with_llm(column_names: list[str], first_rows: list, google_api_key: str = None, openrouter_api_key: str = None) -> dict:
    """
    Analyze columns using LLM and return mapping.
    Can be imported by other modules.
    Tries Google Gemini first, falls back to OpenRouter if it fails.
    """
    if google_api_key is None:
        google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if openrouter_api_key is None:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Try Google Gemini first
    if google_api_key:
        try:
            print("Trying Google Gemini API...")
            mapping = call_google_gemini_api(column_names, first_rows, google_api_key)
            mapping = validate_mapping(mapping, column_names)
            print("✅ Google Gemini API succeeded")
            return mapping
        except Exception as e:
            print(f"⚠️ Google Gemini API failed: {str(e)[:100]}")
            print("Falling back to OpenRouter...")
    
    # Fallback to OpenRouter
    if not openrouter_api_key:
        raise ValueError("Both GOOGLE_API_KEY and OPENROUTER_API_KEY are not set")
    
    print("Using OpenRouter API...")
    mapping = call_openrouter_api(column_names, first_rows, openrouter_api_key)
    mapping = validate_mapping(mapping, column_names)
    print("✅ OpenRouter API succeeded")
    return mapping


def main():
    """Main function to orchestrate the mapping process."""
    
    # Get CSV path from user
    csv_path = input("Enter the path to your CSV file: ").strip()
    
    # Load CSV data
    print(f"Loading CSV from: {csv_path}")
    column_names, first_rows = load_csv_data(csv_path)
    print(f"Found {len(column_names)} columns")
    
    # Get API keys
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Call wrapper function (tries Gemini first, falls back to OpenRouter)
    print("Analyzing columns with LLM...")
    mapping = analyze_columns_with_llm(column_names, first_rows, google_api_key, openrouter_api_key)
    
    # Print final result
    print("\nFinal Mapping:")
    print(json.dumps(mapping, indent=2))
    
    # Save mapping to file for use by cleaning.ipynb
    output_file = os.path.join(os.path.dirname(__file__), "..", "..", "column_mapping.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"\nMapping saved to: {output_file}")
    
    return mapping


if __name__ == "__main__":
    main()