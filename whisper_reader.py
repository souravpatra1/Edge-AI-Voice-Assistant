# whisper_reader.py
import os
import queue

speech_queue = queue.Queue()
PIPE_PATH = "/tmp/whisper_pipe"

def whisper_reader():
    """Continuously reads transcribed text from the FIFO pipe."""
    if not os.path.exists(PIPE_PATH):
        os.mkfifo(PIPE_PATH)

    # Opening in 'r' mode blocks until whisper_service.sh opens it for writing
    with open(PIPE_PATH, "r") as pipe:
        for line in pipe:
            line = line.strip()
            # Ignore system log prefixes output by whisper.cpp
            if not line or line.startswith(("init:", "whisper_", "main:", "ggml_")):
                continue
            speech_queue.put(line)
