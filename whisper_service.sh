#!/bin/bash

WHISPER_BIN="/home/spatra/project/whisper/whisper.cpp/build/bin/whisper-stream"
WHISPER_MODEL="/home/spatra/project/whisper/whisper.cpp/models/ggml-tiny.en.bin"

STEP=4000
LENGTH=8000
KEEP=400
CARD=0
THREADS=2
VAD_THOLD=0.60
MAX_TOKENS=32

PIPE="/tmp/whisper_pipe"
if [ ! -p "$PIPE" ]; then
  mkfifo "$PIPE"
fi

echo "🎙️ Launching Whisper service on CPU cores 0 & 1..."
echo "Output stream writing to $PIPE"

exec taskset -c 0,1 "$WHISPER_BIN" \
  -m "$WHISPER_MODEL" \
  --step "$STEP" \
  --length "$LENGTH" \
  --keep "$KEEP" \
  -c "$CARD" \
  -t "$THREADS" \
  -vth "$VAD_THOLD" \
  --max-tokens "$MAX_TOKENS" \
  -ng \
  --no-fallback > "$PIPE"
