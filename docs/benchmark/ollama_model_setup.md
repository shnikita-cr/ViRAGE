# Ollama модели для benchmark ViRAGE

## Локальные модели основного сравнения

```powershell
ollama pull gemma3:4b
ollama pull qwen2.5vl:3b
ollama pull qwen3-vl:4b
ollama pull llava-phi3:3.8b
ollama pull minicpm-v:latest
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text:latest
ollama pull mxbai-embed-large:latest
ollama pull bge-m3:latest
```

## Облачные модели только для VLM judge control

```powershell
ollama pull gemma4:31b-cloud
ollama pull qwen3.5:cloud
ollama pull qwen3-vl:235b-cloud
```

Если модель на текущем тарифе Ollama Cloud недоступна, она исключается из эксперимента и фиксируется как `missing_models`.

## Hugging Face image-text embeddings

```powershell
hf download openai/clip-vit-base-patch32
hf download google/siglip-so400m-patch14-384
```

## Проверка доступности

```powershell
python scripts\benchmark\model_checks\ping_models.py --device cuda --hf-dtype float16
```

Для CPU-проверки image-text моделей:

```powershell
python scripts\benchmark\model_checks\ping_models.py --skip-ollama-chat --skip-ollama-embeddings --device cpu --hf-dtype float32
```
