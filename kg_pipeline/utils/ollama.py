import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

def check_ollama_server():
    """Checks if the Ollama server is running."""
    try:
        # We check the base URL, not the generate endpoint
        base_url = OLLAMA_API_URL.replace("/api/generate", "")
        response = requests.get(base_url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def call_ollama(prompt: str, model: str = "llama3.1:8b") -> str:
    """
    Calls the Ollama API to generate a response.
    """
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        raise ValueError(f"Failed to get response from Ollama API: {e}")
