```bash
docker exec -it ollama ollama pull qwen3.5:latest
docker exec -it ollama ollama pull qwen3.5:4b
docker exec -it ollama ollama pull gemma3:4b
docker exec -it ollama ollama pull qwen2.5vl:latest
docker exec -it ollama ollama pull gemma3:12b-it-q4_K_M

docker exec -it ollama ollama pull deepseek-r1:8b
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama pull hermes3:8b

docker exec -it ollama ollama pull qwen2.5-coder:7b
docker exec -it ollama ollama pull qwen2.5-coder:3b
docker exec -it ollama ollama pull deepseek-coder:6.7b
docker exec -it ollama ollama pull granite-code:8b
docker exec -it ollama ollama pull codegemma:7b

docker exec -it ollama ollama pull minicpm-v:latest
docker exec -it ollama ollama pull openbmb/minicpm-v4.5:latest
docker exec -it ollama ollama pull moondream:latest

docker exec -it ollama ollama pull nomic-embed-text:latest
docker exec -it ollama ollama pull mxbai-embed-large:latest
docker exec -it ollama ollama pull bge-m3:latest
```

```bash
huggingface-cli download openai/clip-vit-base-patch32
huggingface-cli download openai/clip-vit-large-patch14
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K
huggingface-cli download google/siglip-so400m-patch14-384
huggingface-cli download jinaai/jina-clip-v2
```

