import argparse
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx
import ollama
from PIL import Image, ImageDraw
from ollama import ResponseError


@dataclass(frozen=True)
class ChatPrompt:
    system: str
    user: str
    expects_json: bool
    uses_image: bool


@dataclass(frozen=True)
class BenchmarkResult:
    group: str
    model: str
    latency_seconds: float | None
    throughput: float | None
    status: str
    details: str


OLLAMA_CHAT_COMPONENTS: Mapping[str, Sequence[str]] = {
    "All-in-one": (
        "qwen3.5:latest",
        "qwen3.5:4b",
        "gemma3:4b",
        "qwen2.5vl:latest",
        "gemma3:12b-it-q4_K_M",
    ),
    "Планирование": (
        "qwen3.5:latest",
        "deepseek-r1:8b",
        "gemma3:4b",
        "llama3.1:8b",
        "hermes3:8b",
    ),
    "Генерация Vega-Lite / JSON": (
        "qwen2.5-coder:7b",
        "qwen2.5-coder:3b",
        "deepseek-coder:6.7b",
        "granite-code:8b",
        "codegemma:7b",
    ),
    "VLM": (
        "qwen2.5vl:latest",
        "minicpm-v:latest",
        "openbmb/minicpm-v4.5:latest",
        "gemma3:12b-it-q4_K_M",
        "moondream:latest",
    ),
}

OLLAMA_EMBEDDING_MODELS: Sequence[str] = (
    "nomic-embed-text:latest",
    "mxbai-embed-large:latest",
    "bge-m3:latest",
)

HF_IMAGE_TEXT_MODELS: Sequence[str] = (
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "google/siglip-so400m-patch14-384",
    "jinaai/jina-clip-v2",
)

PROMPTS: Mapping[str, ChatPrompt] = {
    "All-in-one": ChatPrompt(
        system=(
            "You are a strict data visualization agent. Return only valid JSON. "
            "Select one chart action for the user request."
        ),
        user=(
            "Visualize sales by category in descending order as a bar chart. "
            "Return JSON with action and parameters."
        ),
        expects_json=True,
        uses_image=False,
    ),
    "Планирование": ChatPrompt(
        system=(
            "You are an analysis planner. Return only valid JSON with keys: "
            "tasks, skipped_candidates. Each task must include task_id, intent, "
            "required_fields, ranking_strategy, scale_strategy."
        ),
        user=(
            "Find problematic images by brightness, sharpness, contrast, and IQA metrics. "
            "Plan no more than three analytical tasks."
        ),
        expects_json=True,
        uses_image=False,
    ),
    "Генерация Vega-Lite / JSON": ChatPrompt(
        system=(
            "You are a Vega-Lite generator. Return only a strict valid Vega-Lite JSON object. "
            "Do not use markdown fences."
        ),
        user=(
            "Build a bar chart for data [{'x':'A','y':10},{'x':'B','y':20}]. "
            "Use x as nominal X axis and y as quantitative Y axis."
        ),
        expects_json=True,
        uses_image=False,
    ),
    "VLM": ChatPrompt(
        system=(
            "You are a visual chart judge. Inspect the chart image and return only valid JSON "
            "with keys: axes_visible, rendering_errors, readable, recommendation."
        ),
        user="Check whether the chart has visible X and Y axes and rendering errors.",
        expects_json=True,
        uses_image=True,
    ),
}

TASK_TEXT_EN = (
    "A chart image that should show a simple bar chart with visible X and Y axes, "
    "readable labels, and no rendering errors."
)


def clean_json_response(text: str) -> str:
    cleaned = remove_thinking_block(text).strip()
    cleaned = remove_markdown_fence(cleaned)
    return cleaned.strip()


def remove_thinking_block(text: str) -> str:
    if "</think>" not in text:
        return text
    return text.split("</think>", maxsplit=1)[1]


def remove_markdown_fence(text: str) -> str:
    if text.startswith("```json"):
        text = text.removeprefix("```json")
    elif text.startswith("```"):
        text = text.removeprefix("```")
    if text.endswith("```"):
        text = text.removesuffix("```")
    return text


def create_mock_chart_png() -> bytes:
    image = Image.new("RGB", (640, 420), color="white")
    draw = ImageDraw.Draw(image)
    draw.line([(80, 340), (560, 340)], fill="black", width=3)
    draw.line([(80, 80), (80, 340)], fill="black", width=3)
    draw.rectangle([150, 220, 230, 340], fill="steelblue")
    draw.rectangle([310, 150, 390, 340], fill="steelblue")
    draw.text((175, 355), "A", fill="black")
    draw.text((335, 355), "B", fill="black")
    draw.text((40, 70), "Y", fill="black")
    draw.text((565, 330), "X", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run_ollama_chat_group(group_name: str, models: Sequence[str], image_bytes: bytes) -> list[BenchmarkResult]:
    prompt = PROMPTS[group_name]
    return [run_ollama_chat_model(group_name, model, prompt, image_bytes) for model in models]


def run_ollama_chat_model(
    group_name: str,
    model: str,
    prompt: ChatPrompt,
    image_bytes: bytes,
) -> BenchmarkResult:
    messages = build_messages(prompt, image_bytes)
    started_at = time.perf_counter()
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.1},
        )
    except (ResponseError, httpx.HTTPError, ConnectionError, TimeoutError, OSError) as error:
        return failed_result(group_name, model, time.perf_counter() - started_at, error)

    latency = time.perf_counter() - started_at
    content = str(response["message"]["content"])
    status, details = validate_chat_output(content, prompt.expects_json)
    return BenchmarkResult(
        group=group_name,
        model=model,
        latency_seconds=latency,
        throughput=compute_throughput(response, latency),
        status=status,
        details=details,
    )


def build_messages(prompt: ChatPrompt, image_bytes: bytes) -> list[dict[str, Any]]:
    user_message: dict[str, Any] = {"role": "user", "content": prompt.user}
    if prompt.uses_image:
        user_message["images"] = [image_bytes]
    return [
        {"role": "system", "content": prompt.system},
        user_message,
    ]


def validate_chat_output(content: str, expects_json: bool) -> tuple[str, str]:
    if not expects_json:
        return "ok", "response received"
    cleaned = clean_json_response(content)
    try:
        json.loads(cleaned)
    except json.JSONDecodeError as error:
        return "invalid_json", f"{error.msg} at position {error.pos}"
    return "valid_json", "valid JSON"


def run_ollama_embedding_model(model: str) -> BenchmarkResult:
    started_at = time.perf_counter()
    try:
        response = ollama.embed(model=model, input=TASK_TEXT_EN)
    except (ResponseError, httpx.HTTPError, ConnectionError, TimeoutError, OSError) as error:
        return failed_result("RAG-эмбеддинги", model, time.perf_counter() - started_at, error)

    latency = time.perf_counter() - started_at
    embedding_size = get_embedding_size(response)
    return BenchmarkResult(
        group="RAG-эмбеддинги",
        model=model,
        latency_seconds=latency,
        throughput=None,
        status="ok",
        details=f"embedding_dim={embedding_size}",
    )


def get_embedding_size(response: Mapping[str, Any]) -> int:
    embeddings = response.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise ValueError("Ollama embedding response does not contain embeddings")
    first_embedding = embeddings[0]
    if not isinstance(first_embedding, list):
        raise ValueError("Ollama embedding vector has invalid type")
    return len(first_embedding)


def run_hf_image_text_model(model_name: str, image_bytes: bytes, device: str, dtype_name: str) -> BenchmarkResult:
    started_at = time.perf_counter()
    try:
        score = compute_hf_image_text_cosine(model_name, image_bytes, TASK_TEXT_EN, device, dtype_name)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return failed_result("Image-text cosine", model_name, time.perf_counter() - started_at, error)

    return BenchmarkResult(
        group="Image-text cosine",
        model=model_name,
        latency_seconds=time.perf_counter() - started_at,
        throughput=None,
        status="ok",
        details=f"cosine={score:.4f}",
    )


def compute_hf_image_text_cosine(
    model_name: str,
    image_bytes: bytes,
    text: str,
    device: str,
    dtype_name: str,
) -> float:
    import torch
    from transformers import AutoModel, AutoProcessor

    torch_dtype = get_torch_dtype(torch, dtype_name)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        image_features = encode_hf_image(model, processor, image, device)
        text_features = encode_hf_text(model, processor, text, device)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return float((image_features * text_features).sum(dim=-1).item())


def get_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def encode_hf_image(model: Any, processor: Any, image: Image.Image, device: str) -> Any:
    if hasattr(model, "get_image_features"):
        inputs = processor(images=[image], return_tensors="pt").to(device)
        return model.get_image_features(**inputs)
    if hasattr(model, "encode_image"):
        return model.encode_image([image])
    raise ValueError(f"Model {model.__class__.__name__} does not expose image features")


def encode_hf_text(model: Any, processor: Any, text: str, device: str) -> Any:
    if hasattr(model, "get_text_features"):
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        return model.get_text_features(**inputs)
    if hasattr(model, "encode_text"):
        return model.encode_text([text])
    raise ValueError(f"Model {model.__class__.__name__} does not expose text features")


def failed_result(group_name: str, model: str, latency: float, error: BaseException) -> BenchmarkResult:
    return BenchmarkResult(
        group=group_name,
        model=model,
        latency_seconds=latency,
        throughput=None,
        status="failed",
        details=f"{error.__class__.__name__}: {str(error)[:160]}",
    )


def compute_throughput(response: Mapping[str, Any], latency_seconds: float) -> float | None:
    eval_count = response.get("eval_count")
    if not isinstance(eval_count, int) or eval_count <= 0:
        return None
    if latency_seconds <= 0:
        return None
    return eval_count / latency_seconds


def print_result(result: BenchmarkResult) -> None:
    latency = format_latency(result.latency_seconds)
    throughput = format_throughput(result.throughput)
    print(f"{result.group:<32} | {result.model:<40} | {latency:<10} | {throughput:<10} | {result.status:<14} | {result.details}")


def print_header() -> None:
    print("=" * 150)
    print(f"{'Группа':<32} | {'Модель':<40} | {'Время':<10} | {'Ток/сек':<10} | {'Статус':<14} | Детали")
    print("=" * 150)


def format_latency(latency_seconds: float | None) -> str:
    if latency_seconds is None:
        return "n/a"
    return f"{latency_seconds:.2f}s"


def format_throughput(throughput: float | None) -> str:
    if throughput is None:
        return "n/a"
    return f"{throughput:.1f}"


def write_report(results: Sequence[BenchmarkResult], output_path: Path) -> None:
    payload = [
        {
            "group": result.group,
            "model": result.model,
            "latency_seconds": result.latency_seconds,
            "throughput": result.throughput,
            "status": result.status,
            "details": result.details,
        }
        for result in results
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ollama-chat", action="store_true")
    parser.add_argument("--skip-ollama-embeddings", action="store_true")
    parser.add_argument("--skip-hf-cosine", action="store_true")
    parser.add_argument("--hf-device", default="cuda")
    parser.add_argument("--hf-dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_component_smoke/report.json"))
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    image_bytes = create_mock_chart_png()
    results: list[BenchmarkResult] = []

    if not args.skip_ollama_chat:
        for group_name, models in OLLAMA_CHAT_COMPONENTS.items():
            results.extend(run_ollama_chat_group(group_name, models, image_bytes))

    if not args.skip_ollama_embeddings:
        results.extend(run_ollama_embedding_model(model) for model in OLLAMA_EMBEDDING_MODELS)

    if not args.skip_hf_cosine:
        results.extend(
            run_hf_image_text_model(model, image_bytes, args.hf_device, args.hf_dtype)
            for model in HF_IMAGE_TEXT_MODELS
        )

    return results


def main() -> None:
    args = parse_args()
    results = run_benchmark(args)
    print_header()
    for result in results:
        print_result(result)
    print("=" * 150)
    write_report(results, args.output)


if __name__ == "__main__":
    main()
