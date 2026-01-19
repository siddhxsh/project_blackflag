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
        "model": "openai/gpt-4o-mini",
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


def analyze_columns_with_llm(column_names: list[str], first_rows: list, api_key: str = None) -> dict:
    """
    Analyze columns using LLM and return mapping.
    Can be imported by other modules.
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        # Use hardcoded key for deployment
        api_key = "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"
    
    # Call API
    mapping = call_openrouter_api(column_names, first_rows, api_key)
    
    # Validate mapping
    mapping = validate_mapping(mapping, column_names)
    
    return mapping


def main():
    """Main function to orchestrate the mapping process."""
    
    # Get CSV path from user
    csv_path = input("Enter the path to your CSV file: ").strip()
    
    # Load CSV data
    print(f"Loading CSV from: {csv_path}")
    column_names, first_rows = load_csv_data(csv_path)
    print(f"Found {len(column_names)} columns")
    
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Call wrapper function
    print("Calling OpenRouter API for column mapping...")
    mapping = analyze_columns_with_llm(column_names, first_rows, api_key)
    
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