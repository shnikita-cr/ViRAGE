import argparse
import io
import time
from typing import Any, Mapping, Sequence
import ollama
from PIL import Image

OLLAMA_MODELS: Mapping[str, Sequence[str]] = {
    "All-in-one / Reasoning / Code / VLM": (
        "qwen3.5:latest", "qwen3.5:4b", "gemma3:4b", "qwen2.5vl:latest",
        "gemma3:12b-it-q4_K_M", "deepseek-r1:8b", "llama3.1:8b", "hermes3:8b",
        "qwen2.5-coder:7b", "qwen2.5-coder:3b", "deepseek-coder:6.7b",
        "granite-code:8b", "codegemma:7b", "minicpm-v:latest",
        "openbmb/minicpm-v4.5:latest", "moondream:latest"
    ),
    "Embeddings": (
        "nomic-embed-text:latest", "mxbai-embed-large:latest", "bge-m3:latest"
    )
}

HF_MODELS: Sequence[str] = (
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "google/siglip-so400m-patch14-384",
    "jinaai/jina-clip-v2"
)


def get_mock_img() -> bytes:
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_ollama_llm(model: str, img: bytes) -> tuple[float, float | str]:
    use_img = any(x in model.lower() for x in ["vl", "v4.5", "minicpm", "moondream", "gemma3"])
    msg = {"role": "user", "content": "Hi"}
    if use_img:
        msg["images"] = [img]

    try:
        # 1. Прогревочный запрос (загрузка весов в VRAM)
        ollama.chat(model=model, messages=[msg], options={"num_predict": 1})

        # 2. Повторный замер чистого инференса
        start = time.perf_counter()
        res = ollama.chat(model=model, messages=[msg], options={"num_predict": 15, "num_ctx": 2048})
        latency = time.perf_counter() - start

        tokens = res.get("eval_count") if isinstance(res, dict) else getattr(res, "eval_count", 0)
        tps = float(tokens) / latency if tokens and latency > 0 else 0.0
        return latency, tps
    except Exception as e:
        return 0.0, f"Error: {type(e).__name__}"


def run_ollama_embed(model: str) -> tuple[float, str]:
    try:
        # 1. Прогревочный запрос
        ollama.embed(model=model, input="Warmup")

        # 2. Чистый замер
        start = time.perf_counter()
        ollama.embed(model=model, input="Benchmark")
        return time.perf_counter() - start, "N/A (Embedding)"
    except Exception as e:
        return 0.0, f"Error: {type(e).__name__}"


def run_hf_model(model_name: str, img_bytes: bytes, device: str) -> tuple[float, str]:
    import torch
    from transformers import AutoModel, AutoProcessor

    try:
        # 1. Инициализация и прогревочный проход (Перенос на GPU/загрузка кэша)
        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if "siglip" in model_name.lower():
            inputs = proc(text=["test"], images=image, return_tensors="pt", padding="max_length")
        else:
            inputs = proc(text=["test"], images=image, return_tensors="pt", padding=True)

        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            model(**inputs)  # прогрев

            # 2. Чистый повторный замер выполнения тензорных операций
            start = time.perf_counter()
            outputs = model(**inputs)
            if hasattr(outputs, "image_embeds"):
                _ = outputs.image_embeds
            elif hasattr(model, "get_image_features"):
                _ = model.get_image_features(**inputs)
            latency = time.perf_counter() - start

        return latency, "N/A (Encoder)"
    except Exception as e:
        return 0.0, f"Error: {type(e).__name__}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="cuda, cpu, mps")
    args = parser.parse_args()

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("⚠️ Найдена ошибка: PyTorch не видит CUDA! Переключаюсь на CPU для HF.")
            args.device = "cpu"

    img = get_mock_img()
    print(f"\n{'Тип':<12} | {'Модель':<40} | {'Ping VRAM (с)':<13} | {'Чистая скорость (ток/с)'}")
    print("-" * 90)

    for model in OLLAMA_MODELS["All-in-one / Reasoning / Code / VLM"]:
        lat, tps = run_ollama_llm(model, img)
        tps_str = f"{tps:.1f}" if isinstance(tps, float) else tps
        print(f"{'Ollama LLM':<12} | {model:<40} | {lat:.3f}        | {tps_str}")

    for model in OLLAMA_MODELS["Embeddings"]:
        lat, tps = run_ollama_embed(model)
        print(f"{'Ollama Emb':<12} | {model:<40} | {lat:.3f}        | {tps}")

    print("\nЗагрузка и проверка Hugging Face моделей...")
    for model in HF_MODELS:
        lat, tps = run_hf_model(model, img, args.device)
        print(f"{'HF CLIP':<12} | {model:<40} | {lat:.3f}        | {tps}")


if __name__ == "__main__":
    main()
