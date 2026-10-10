# ai_pipeline.py
import os
import threading
from whisper_reader import whisper_reader
from ai_consumer import ai_consumer

def main():
    # Pin Python pipeline and Ollama inference callers to Cores 1, 2, 3
    try:
        os.sched_setaffinity(0, {1, 2, 3})
        print("🧠 Python & LLM consumer pinned to CPU cores 1, 2 & 3")
    except AttributeError:
        pass

    print("🎙️ Reading from Whisper service stream...")

    # Start reader and consumer threads
    threading.Thread(target=whisper_reader, daemon=True).start()
    threading.Thread(target=ai_consumer, daemon=True).start()

    # Keep main thread alive
    threading.Event().wait()

if __name__ == "__main__":
    main()
