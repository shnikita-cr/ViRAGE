### All-in-one: планирование + VLM + простая генерация JSON

| Модель                 | Источник | Размер / вес | Вход       |
| ---------------------- | -------- | -----------: | ---------- |
| `qwen3.5:latest`       | Ollama   |      ~6.6 GB | Text/Image |
| `qwen3.5:4b`           | Ollama   |      ~3.4 GB | Text/Image |
| `gemma3:4b`            | Ollama   |      ~3.3 GB | Text/Image |
| `qwen2.5vl:3b`     | Ollama   |      ~6.0 GB | Text/Image |
| `gemma3:12b-it-q4_K_M` | Ollama   |      ~8.1 GB | Text/Image |

### Планирование / reasoning

| Модель           | Источник | Размер / вес | Вход       |
| ---------------- | -------- | -----------: | ---------- |
| `qwen3.5:latest` | Ollama   |      ~6.6 GB | Text/Image |
| `deepseek-r1:8b` | Ollama   |      ~5.2 GB | Text       |
| `gemma3:4b`      | Ollama   |      ~3.3 GB | Text/Image |
| `llama3.1:8b`    | Ollama   |      ~4.9 GB | Text       |
| `hermes3:8b`     | Ollama   |      ~4.7 GB | Text       |

### Генерация Vega-Lite / JSON / code

| Модель                | Источник |       Размер / вес | Вход |
| --------------------- | -------- | -----------------: | ---- |
| `qwen2.5-coder:7b`    | Ollama   |            ~4.7 GB | Text |
| `qwen2.5-coder:3b`    | Ollama   |            ~1.9 GB | Text |
| `deepseek-coder:6.7b` | Ollama   |            ~3.8 GB | Text |
| `granite-code:8b`     | Ollama   |            ~4.6 GB | Text |
| `codegemma:7b`        | Ollama   | проверить локально | Text |

### VLM: судья графика + финальный визуальный анализ

| Модель                        | Источник                   | Размер / вес | Вход       |
| ----------------------------- | -------------------------- | -----------: | ---------- |
| `qwen2.5vl:3b`            | Ollama                     |      ~6.0 GB | Text/Image |
| `minicpm-v:latest`            | Ollama                     |      ~5.5 GB | Text/Image |
| `openbmb/minicpm-v4.5:latest` | Ollama / registry-зависимо |      ~6.1 GB | Text/Image |
| `gemma3:12b-it-q4_K_M`        | Ollama                     |      ~8.1 GB | Text/Image |
| `moondream:latest`            | Ollama                     |      ~1.7 GB | Text/Image |

### RAG-эмбеддинги

| Модель                     | Источник | Размер / вес | Вход |
| -------------------------- | -------- | -----------: | ---- |
| `nomic-embed-text:latest`  | Ollama   |      ~274 MB | Text |
| `mxbai-embed-large:latest` | Ollama   |      ~669 MB | Text |
| `bge-m3:latest`            | Ollama   |      ~1.2 GB | Text |

### Image-text cosine: PNG + TaskText

| Модель                                  | Источник     |            Размер / вес | Вход       |
| --------------------------------------- | ------------ | ----------------------: | ---------- |
| `openai/clip-vit-base-patch32`          | Hugging Face | ~0.151B / ~0.30 GB FP16 | Image/Text |
| `openai/clip-vit-large-patch14`         | Hugging Face | ~0.428B / ~0.86 GB FP16 | Image/Text |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | Hugging Face | ~0.986B / ~1.97 GB FP16 | Image/Text |
| `google/siglip-so400m-patch14-384`      | Hugging Face | ~0.900B / ~1.80 GB FP16 | Image/Text |
| `jinaai/jina-clip-v2`                   | Hugging Face | ~0.865B / ~1.73 GB FP16 | Image/Text |
