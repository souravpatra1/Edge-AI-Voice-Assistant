# ai_consumer.py
import re
import json
import queue
import requests
import threading
import subprocess
from whisper_reader import speech_queue
from utils import config
import memory_logger

tts_queue = queue.Queue()
is_speaking = False

def persistent_audio_pipeline():
    """Persistent low-latency TTS worker thread."""
    global is_speaking
    
    piper_proc = subprocess.Popen(
        [config.PIPER_BIN, "--model", config.PIPER_MODEL, "--output_raw", "--json-input"],
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
    return any(re.search(pat, clean_line) for pat in config.HALLUCINATION_PATTERNS)

def is_summary_request(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in config.SUMMARY_KEYWORDS)

def is_addressed_to_ai(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    has_wake_word = any(ww in prompt_lower for ww in config.WAKE_WORDS)
    return has_wake_word or is_summary_request(prompt_lower)

def stream_llm_and_speak(prompt: str):
    """Dynamic LLM handler configured entirely via utils.config."""
    asking_for_summary = is_summary_request(prompt)

    if asking_for_summary:
        daily_context = memory_logger.get_today_transcript()
        formatted_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful meeting assistant. Use the following transcript of today's conversations to summarize or answer the user's question accurately and concisely.\n\n"
            f"--- TODAY'S FULL TRANSCRIPT ---\n{daily_context}\n-------------------------------<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        ctx_size = config.MEMORY_NUM_CTX
        max_tokens = config.MEMORY_NUM_PREDICT
    else:
        formatted_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a concise voice assistant. Give brief, direct answers.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        ctx_size = config.FAST_NUM_CTX
        max_tokens = config.FAST_NUM_PREDICT

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": formatted_prompt,
        "options": {
            "num_thread": config.LLM_THREADS,
            "num_ctx": ctx_size,
            "num_predict": max_tokens,
            "temperature": config.LLM_TEMPERATURE
        }
    }
    
    word_buffer = []
    full_response = ""
    
    try:
        response = requests.post(
            config.OLLAMA_URL, 
            json=payload, 
            stream=True, 
            timeout=config.LLM_TIMEOUT
        )
        
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

        memory_logger.log_speech("AI", full_response.strip())

    except Exception as e:
        print(f"\n[LLM Error]: {e}")

def ai_consumer():
    """Main speech queue consumer loop."""
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
            speech_chunk = " ".join(buffer)
            buffer.clear()

            memory_logger.log_speech("Human", speech_chunk)
            print(f"\n[Recorded]: {speech_chunk}")

            if is_addressed_to_ai(speech_chunk):
                print(f"AI Triggered: ", end="", flush=True)
                stream_llm_and_speak(speech_chunk)
