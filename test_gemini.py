"""Optional Gemini connectivity probe that never prints credentials or response bodies."""

import os

import requests
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Gemini API key is not configured.")
        return 1
    try:
        response = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": api_key},
            timeout=(5, 15),
        )
        print(f"Gemini models endpoint returned HTTP {response.status_code}.")
        return 0 if response.ok else 1
    except requests.RequestException as exc:
        print(f"Gemini request failed ({type(exc).__name__}).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
