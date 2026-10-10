import os
import re
import json
import queue
import requests
import threading
import subprocess
from whisper_reader import speech_queue
from utils import config
import memory_logger

PIPER_BIN = os.path.expanduser("~/project/piper/piper/piper")
PIPER_MODEL = os.path.expanduser("~/project/piper/en_US-lessac-low.onnx")

HALLUCINATION_PATTERNS = [
    r"\[.*\]", r"\(.*\)", r"^thanks for watching$", r"^thank you\.$", r"^you\.$", r"^\.$"
]

# Memory query intent triggers
SUMMARY_KEYWORDS = ["summary", "summarize", "recap", "what did we discuss", "remember", "meeting", "earlier today"]

tts_queue = queue.Queue()
is_speaking = False

def persistent_audio_pipeline():
    """Persistent low-latency TTS worker thread."""
    global is_speaking
    
    piper_proc = subprocess.Popen(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output_raw", "--json-input"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )

    paplay_proc = subprocess.Popen(
        [
            "paplay", 
            "--raw", 
            "--rate=22050", 
            "--channels=1", 
            "--format=s16le",
            "--latency-msec=100"
        ],
        stdin=piper_proc.stdout,
        stderr=subprocess.DEVNULL
    )

    while True:
        text = tts_queue.get()
        if text is None:
            break

        is_speaking = True
        try:
            json_payload = json.dumps({"text": text}) + "\n"
            piper_proc.stdin.write(json_payload)
            piper_proc.stdin.flush()
        except Exception as e:
            print(f"\n[TTS Error]: {e}")
        finally:
            tts_queue.task_done()
            if tts_queue.empty():
                is_speaking = False

threading.Thread(target=persistent_audio_pipeline, daemon=True).start()

def clean_text(text: str) -> str:
    text = re.sub(r"\*|_|#|`|~", "", text)
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    return text.strip()

def is_hallucination(line: str) -> bool:
    clean_line = line.strip().lower()
    if not clean_line or len(clean_line) < 3:
        return True
    return any(re.search(pat, clean_line) for pat in HALLUCINATION_PATTERNS)

def is_summary_request(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in SUMMARY_KEYWORDS)

def stream_llm_and_speak(prompt: str):
    """Dynamic LLM handler switching between fast voice mode and daily memory mode."""
    
    # 1. Log the user speech immediately to memory
    memory_logger.log_speech("User", prompt)

    # 2. Determine if this requires searching daily memory
    asking_for_summary = is_summary_request(prompt)

    if asking_for_summary:
        daily_context = memory_logger.get_today_transcript()
        formatted_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant. Use the following transcript of today's events to answer the user's request concisely.\n\n"
            f"--- TODAY'S TRANSCRIPT ---\n{daily_context}\n--------------------------<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        ctx_size = 4096
        max_tokens = 128
    else:
        # Standard ultra-low latency prompt configuration
        formatted_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a concise voice assistant. Give brief, direct answers.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        ctx_size = 256
        max_tokens = 32

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": formatted_prompt,
        "options": {
            "num_thread": 3,
            "num_ctx": ctx_size,
            "num_predict": max_tokens,
            "temperature": 0.2
        }
    }
    
    word_buffer = []
    full_response = ""
    
    try:
        response = requests.post(config.OLLAMA_URL, json=payload, stream=True, timeout=30)
        
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            token = chunk.get("response", "")
            
            print(token, end="", flush=True)
            full_response += token
            
            if token.strip():
                word_buffer.append(token)

            full_chunk_text = "".join(word_buffer)
            if len(word_buffer) >= 3 or any(p in token for p in [".", "?", "!", ","]):
                cleaned = clean_text(full_chunk_text)
                if cleaned:
                    tts_queue.put(cleaned)
                    word_buffer.clear()

        if word_buffer:
            cleaned = clean_text("".join(word_buffer))
            if cleaned:
                tts_queue.put(cleaned)

        # 3. Log AI response to daily memory
        memory_logger.log_speech("AI", full_response.strip())

    except Exception as e:
        print(f"\n[LLM Error]: {e}")

def ai_consumer():
    global is_speaking
    buffer = []
    
    while True:
        line = speech_queue.get()
        speech_queue.task_done()

        if is_speaking or not tts_queue.empty():
            continue

        line = line.strip()
        if is_hallucination(line):
            continue

        buffer.append(line)

        if line.endswith((".", "?", "!")):
            user_speech = " ".join(buffer)
            buffer.clear()

            print(f"\n\nYou: {user_speech}")
            print("AI: ", end="", flush=True)
            
            stream_llm_and_speak(user_speech)
