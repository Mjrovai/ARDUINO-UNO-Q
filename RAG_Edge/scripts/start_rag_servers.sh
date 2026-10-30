#!/usr/bin/env bash
# Start the two llama-server processes this chapter needs, in the background.
#   SLM (Qwen3.5-0.8B) on port 8081, embedding model (bge-small) on port 8082.
# Logs go to ~/rag/logs; stop both with: pkill llama-server
LLAMA=~/llama.cpp/build/bin/llama-server
SLM=${SLM:-~/models/Qwen_Qwen3.5-0.8B-Q8_0.gguf}
EMB=${EMB:-~/models/embed/bge-small-en-v1.5-q8_0.gguf}
mkdir -p ~/rag/logs

nohup "$LLAMA" --model "$SLM" --port 8081 --host 0.0.0.0 \
  --jinja --reasoning off --reasoning-budget 0 \
  --ctx-size 4096 --threads 4 --batch-size 512 --ubatch-size 64 \
  --predict 1024 --alias qwen3.5-0.8b > ~/rag/logs/slm.log 2>&1 &

nohup "$LLAMA" --model "$EMB" --port 8082 --embedding \
  --ctx-size 2048 --batch-size 2048 --ubatch-size 2048 --threads 4 \
  --alias bge-small > ~/rag/logs/embed.log 2>&1 &

for port in 8081 8082; do
  until curl -s localhost:$port/health | grep -q ok; do sleep 1; done
  echo "port $port ready"
done
