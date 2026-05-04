
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def list_free_models():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env")
        return

    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "5G RCA Assistant",
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        models = response.json().get("data", [])
        
        free_models = [m for m in models if "free" in m.get("id", "").lower() or (m.get("pricing") and float(m["pricing"]["prompt"]) == 0)]
        
        print(f"Found {len(free_models)} free models:")
        for m in free_models:
            print(f"- {m['id']}")
            
    except Exception as e:
        print(f"Error fetching models: {e}")

if __name__ == "__main__":
    list_free_models()
