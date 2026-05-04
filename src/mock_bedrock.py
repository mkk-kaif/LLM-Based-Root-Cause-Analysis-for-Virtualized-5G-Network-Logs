# =============================================================================
# mock_bedrock.py
#
# Purpose:
#   This module simulates responses from a large language model (LLM) such as Bedrock Claude or Titan.
#   It is used for local development and testing of the RCA assistant pipeline without requiring access
#   to an actual LLM API. The mock function returns a fixed response and echoes part of the prompt.
# =============================================================================

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def query_llm(prompt: str, model: str = "openrouter/auto") -> str:
    """
    Queries the OpenRouter API to get a real root cause analysis.
    Uses OpenAI-compatible client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return "[Configuration Error]: OPENAI_API_KEY is not set in .env"

    try:
        # OpenRouter uses the OpenAI client but requires a different base_url
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8501", 
                "X-Title": "5G RCA Assistant", 
            },
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional 5G Core Network Engineer specializing in root cause analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"[OpenRouter API Error]: {str(e)}"
