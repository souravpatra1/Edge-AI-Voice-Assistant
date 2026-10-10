# utils/config.py
import os

# --- Paths & Executables ---
BASE_DIR = os.path.expanduser("~/project")
WHISPER_DIR = os.path.join(BASE_DIR, "whisper/whisper.cpp")
PIPER_DIR = os.path.join(BASE_DIR, "piper")

WHISPER_BIN = os.path.join(WHISPER_DIR, "build/bin/whisper-stream")
WHISPER_MODEL = os.path.join(WHISPER_DIR, "models/ggml-base.en.bin")

PIPER_BIN = os.path.join(PIPER_DIR, "piper/piper")
PIPER_MODEL = os.path.join(PIPER_DIR, "en_US-lessac-low.onnx")

LOG_DIR = os.path.join(BASE_DIR, "daily_transcripts")
WHISPER_PIPE = "/tmp/whisper_pipe"

# --- Audio & STT Settings ---
AUDIO_CARD = 0
VAD_THRESHOLD = 0.50
WHISPER_STEP_MS = 1000
WHISPER_LENGTH_MS = 3000
WHISPER_KEEP_MS = 300
WHISPER_THREADS = 3

# --- LLM Endpoint & Models ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

# --- Latency & Generation Options ---
FAST_NUM_CTX = 256
FAST_NUM_PREDICT = 32
MEMORY_NUM_CTX = 4096
MEMORY_NUM_PREDICT = 128
LLM_TEMPERATURE = 0.2
LLM_THREADS = 3
LLM_TIMEOUT = 30

# --- Keywords & Intent ---
WAKE_WORDS = ["computer", "hello ai", "hey ai", "assistant", "ai"]
SUMMARY_KEYWORDS = [
    "summary", "summarize", "recap", "what did we discuss", 
    "remember", "meeting", "earlier today"
]

# --- Noise / Hallucination Filters ---
HALLUCINATION_PATTERNS = [
    r"\[.*\]", r"\(.*\)", r"^thanks for watching$", r"^thank you\.$", r"^you\.$", r"^\.$"
]
