"""Optional Groq connectivity probe; never prints credentials or response bodies."""

import os

import requests
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Groq API key is not configured.")
        return 1
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=(5, 15),
        )
        model_count = len(response.json().get("data", [])) if response.ok else 0
        print(f"Groq models endpoint returned HTTP {response.status_code}; {model_count} models available.")
        return 0 if response.ok else 1
    except (requests.RequestException, ValueError) as exc:
        print(f"Groq request failed ({type(exc).__name__}).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
