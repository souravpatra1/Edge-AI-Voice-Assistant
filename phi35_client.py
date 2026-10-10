import requests
from utils import config

def phi35_stream(prompt: str):
    """Send prompt to Phi-3.5 with optimized runtime settings."""
    # print(f"[DEBUG] Sending prompt to Phi3.5: {prompt}")
    
    payload = {
        "model": config.PHI35_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_thread": 3,      # Use 3 CPU threads (Cores 1, 2, 3)
            "num_ctx": 1024,      # Lower context window from default 4096 to 1024
            "num_predict": 128,   # Cap generation to keep responses concise
            "temperature": 0.6,
            "repeat_penalty": 1.1
        }
    }

    try:
        with requests.post(config.PHI35_URL, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    chunk = line.decode("utf-8")
                    if '"response":"' in chunk:
                        text = chunk.split('"response":"')[1].split('"')[0]
                        print(text, end="", flush=True)
        print()
    except Exception as e:
        print(f"\n[ERROR] LLM Request failed: {e}")
