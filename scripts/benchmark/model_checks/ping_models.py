from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import io
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import httpx
from PIL import Image, ImageDraw


def _import_ollama() -> tuple[Any, type[BaseException]]:
    import ollama
    from ollama import ResponseError

    return ollama, ResponseError


def _import_torch() -> Any:
    import torch

    return torch

@dataclass(frozen=True)
class PingResult:
    category: str
    model: str
    latency_seconds: float | None
    throughput_tokens_per_second: float | None
    status: str
    details: str
OLLAMA_CHAT_MODELS: Mapping[str, Sequence[str]] = {'All-in-one / Reasoning / Code / VLM': ('qwen3.5:latest', 'qwen3.5:4b', 'gemma3:4b', 'qwen2.5vl:latest', 'gemma3:12b-it-q4_K_M', 'deepseek-r1:8b', 'llama3.1:8b', 'hermes3:8b', 'qwen2.5-coder:7b', 'qwen2.5-coder:3b', 'deepseek-coder:6.7b', 'granite-code:8b', 'codegemma:7b', 'minicpm-v:latest', 'openbmb/minicpm-v4.5:latest', 'moondream:latest')}
OLLAMA_EMBEDDING_MODELS: Sequence[str] = ('nomic-embed-text:latest', 'mxbai-embed-large:latest', 'bge-m3:latest')
HF_IMAGE_TEXT_MODELS: Sequence[str] = ('openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14', 'google/siglip-so400m-patch14-384', 'jinaai/jina-clip-v2')
IMAGE_CAPABLE_OLLAMA_MARKERS: Sequence[str] = ('vl', 'minicpm', 'moondream', 'gemma3')

def create_mock_chart_png() -> bytes:
    image = Image.new('RGB', (320, 240), color='white')
    draw = ImageDraw.Draw(image)
    draw.line([(40, 200), (280, 200)], fill='black', width=2)
    draw.line([(40, 40), (40, 200)], fill='black', width=2)
    draw.rectangle([80, 130, 120, 200], fill='steelblue')
    draw.rectangle([170, 80, 210, 200], fill='steelblue')
    draw.text((95, 210), 'A', fill='black')
    draw.text((185, 210), 'B', fill='black')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()

def model_accepts_image(model: str) -> bool:
    normalized_model = model.lower()
    return any((marker in normalized_model for marker in IMAGE_CAPABLE_OLLAMA_MARKERS))

def run_ollama_chat(model: str, image_bytes: bytes, num_ctx: int, num_predict: int) -> PingResult:
    try:
        ollama, response_error = _import_ollama()
    except ImportError as error:
        return failed_result('Ollama LLM', model, error)
    message = {'role': 'user', 'content': 'Return one short JSON object with key status and value ok.'}
    if model_accepts_image(model):
        message['images'] = [image_bytes]
    options = {'num_predict': num_predict, 'num_ctx': num_ctx, 'temperature': 0.1}
    warmup_options = {'num_predict': 1, 'num_ctx': num_ctx, 'temperature': 0.1}
    try:
        ollama.chat(model=model, messages=[message], options=warmup_options)
        started_at = time.perf_counter()
        response = ollama.chat(model=model, messages=[message], options=options)
    except (response_error, httpx.HTTPError, ConnectionError, TimeoutError, OSError) as error:
        return failed_result('Ollama LLM', model, error)
    latency = time.perf_counter() - started_at
    throughput = compute_ollama_throughput(response, latency)
    return PingResult('Ollama LLM', model, latency, throughput, 'ok', 'chat completed')

def run_ollama_embedding(model: str) -> PingResult:
    try:
        ollama, response_error = _import_ollama()
    except ImportError as error:
        return failed_result('Ollama Emb', model, error)
    try:
        ollama.embed(model=model, input='Warmup text for embedding ping.')
        started_at = time.perf_counter()
        response = ollama.embed(model=model, input='Benchmark text for embedding ping.')
    except (response_error, httpx.HTTPError, ConnectionError, TimeoutError, OSError) as error:
        return failed_result('Ollama Emb', model, error)
    latency = time.perf_counter() - started_at
    embedding_dimension = extract_embedding_dimension(response)
    return PingResult('Ollama Emb', model, latency, None, 'ok', f'embedding_dim={embedding_dimension}')

def run_hf_image_text(model_name: str, image_bytes: bytes, device: str, dtype_name: str) -> PingResult:
    started_at = time.perf_counter()
    try:
        score = compute_hf_image_text_cosine(model_name, image_bytes, device, dtype_name)
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError, TypeError) as error:
        return failed_result('HF cosine', model_name, error)
    latency = time.perf_counter() - started_at
    return PingResult('HF cosine', model_name, latency, None, 'ok', f'cosine={score:.4f}')

def compute_hf_image_text_cosine(model_name: str, image_bytes: bytes, device: str, dtype_name: str) -> float:
    torch = _import_torch()
    from transformers import AutoModel, AutoProcessor
    if device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('PyTorch CUDA is not available in this environment')
    torch_dtype = resolve_torch_dtype(dtype_name)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, dtype=torch_dtype).to(device)
    model.eval()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    text = 'A simple readable bar chart with visible axes.'
    with torch.no_grad():
        image_features = compute_image_features(model, processor, image, device)
        text_features = compute_text_features(model, processor, text, device)
        image_features = normalize_features(image_features)
        text_features = normalize_features(text_features)
        return float((image_features * text_features).sum(dim=-1).item())

def compute_image_features(model: Any, processor: Any, image: Image.Image, device: str) -> Any:
    torch = _import_torch()
    inputs = processor(images=image, return_tensors='pt')
    inputs = move_tensor_mapping_to_device(inputs, device)
    if hasattr(model, 'get_image_features'):
        return extract_feature_tensor(model.get_image_features(**inputs), feature_name='image')
    if hasattr(model, 'encode_image'):
        if 'pixel_values' not in inputs:
            raise ValueError('Processor output does not contain pixel_values for image encoding')
        return extract_feature_tensor(model.encode_image(inputs['pixel_values']), feature_name='image')
    return extract_feature_tensor(model(**inputs), feature_name='image')

def compute_text_features(model: Any, processor: Any, text: str, device: str) -> Any:
    torch = _import_torch()
    inputs = processor(text=[text], return_tensors='pt', padding=True, truncation=True)
    inputs = move_tensor_mapping_to_device(inputs, device)
    inputs.pop('token_type_ids', None)
    if hasattr(model, 'get_text_features'):
        return extract_feature_tensor(model.get_text_features(**inputs), feature_name='text')
    if hasattr(model, 'encode_text'):
        if 'input_ids' not in inputs:
            raise ValueError('Processor output does not contain input_ids for text encoding')
        return extract_feature_tensor(model.encode_text(inputs['input_ids']), feature_name='text')
    return extract_feature_tensor(model(**inputs), feature_name='text')

def extract_feature_tensor(output: Any, feature_name: str) -> Any:
    torch = _import_torch()
    if isinstance(output, torch.Tensor):
        return ensure_two_dimensional(output)
    candidate_attribute_names = get_candidate_attribute_names(feature_name)
    for attribute_name in candidate_attribute_names:
        value = getattr(output, attribute_name, None)
        if isinstance(value, torch.Tensor):
            return ensure_two_dimensional(value)
    last_hidden_state = getattr(output, 'last_hidden_state', None)
    if isinstance(last_hidden_state, torch.Tensor):
        return mean_pool_last_hidden_state(last_hidden_state)
    if isinstance(output, Mapping):
        for key in candidate_attribute_names:
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return ensure_two_dimensional(value)
        hidden_state = output.get('last_hidden_state')
        if isinstance(hidden_state, torch.Tensor):
            return mean_pool_last_hidden_state(hidden_state)
    raise ValueError(f'Unable to extract {feature_name} feature tensor from {output.__class__.__name__}')

def get_candidate_attribute_names(feature_name: str) -> tuple[str, ...]:
    if feature_name == 'image':
        return ('image_embeds', 'image_features', 'pooler_output', 'embeds')
    if feature_name == 'text':
        return ('text_embeds', 'text_features', 'pooler_output', 'embeds')
    raise ValueError(f'Unsupported feature name: {feature_name}')

def ensure_two_dimensional(features: Any) -> Any:
    if features.ndim == 1:
        return features.unsqueeze(0)
    if features.ndim == 2:
        return features
    if features.ndim == 3:
        return features.mean(dim=1)
    raise ValueError(f'Unsupported feature tensor dimensions: {features.ndim}')

def mean_pool_last_hidden_state(last_hidden_state: Any) -> Any:
    if last_hidden_state.ndim != 3:
        raise ValueError(f'Unsupported last_hidden_state dimensions: {last_hidden_state.ndim}')
    return last_hidden_state.mean(dim=1)

def move_tensor_mapping_to_device(inputs: Mapping[str, Any], device: str) -> dict[str, Any]:
    torch = _import_torch()
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

def normalize_features(features: Any) -> Any:
    torch = _import_torch()
    features = features.float()
    norm = features.norm(dim=-1, keepdim=True)
    if torch.any(norm == 0):
        raise ValueError('Feature tensor contains zero-norm vectors')
    return features / norm

def resolve_torch_dtype(dtype_name: str) -> Any:
    torch = _import_torch()
    if dtype_name == 'float16':
        return torch.float16
    if dtype_name == 'bfloat16':
        return torch.bfloat16
    if dtype_name == 'float32':
        return torch.float32
    raise ValueError(f'Unsupported dtype: {dtype_name}')

def extract_embedding_dimension(response: Mapping[str, Any]) -> int:
    embeddings = response.get('embeddings')
    if not isinstance(embeddings, list) or not embeddings:
        raise ValueError('Ollama embedding response does not contain embeddings')
    embedding = embeddings[0]
    if not isinstance(embedding, list):
        raise ValueError('Ollama embedding vector has invalid type')
    return len(embedding)

def compute_ollama_throughput(response: Mapping[str, Any], latency_seconds: float) -> float | None:
    eval_count = response.get('eval_count')
    if not isinstance(eval_count, int) or eval_count <= 0:
        return None
    if latency_seconds <= 0:
        return None
    return eval_count / latency_seconds

def failed_result(category: str, model: str, error: BaseException) -> PingResult:
    return PingResult(category=category, model=model, latency_seconds=None, throughput_tokens_per_second=None, status='failed', details=f'{error.__class__.__name__}: {str(error)[:260]}')

def print_torch_status(device: str) -> None:
    try:
        torch = _import_torch()
    except ImportError as error:
        logger.info(f'PyTorch: unavailable ({error.__class__.__name__}: {error})')
        logger.info(f'Requested HF device: {device}')
        logger.info('')
        return
    logger.info(f'PyTorch: {torch.__version__}')
    logger.info(f'Requested HF device: {device}')
    logger.info(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logger.info(f'CUDA version: {torch.version.cuda}')
        logger.info(f'CUDA device: {torch.cuda.get_device_name(0)}')
    logger.info('')

def print_result(result: PingResult) -> None:
    latency = format_latency(result.latency_seconds)
    throughput = format_throughput(result.throughput_tokens_per_second)
    logger.info(f'{result.category:<12} | {result.model:<42} | {latency:<10} | {throughput:<10} | {result.status:<8} | {result.details}')

def format_latency(latency_seconds: float | None) -> str:
    if latency_seconds is None:
        return 'n/a'
    return f'{latency_seconds:.3f}s'

def format_throughput(tokens_per_second: float | None) -> str:
    if tokens_per_second is None:
        return 'n/a'
    return f'{tokens_per_second:.1f}'

def print_header() -> None:
    logger.info(f"{'Тип':<12} | {'Модель':<42} | {'Ping':<10} | {'Ток/сек':<10} | {'Статус':<8} | Детали")
    logger.info('-' * 126)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--hf-dtype', default='float16', choices=('float16', 'bfloat16', 'float32'))
    parser.add_argument('--num-ctx', type=int, default=2048)
    parser.add_argument('--num-predict', type=int, default=15)
    parser.add_argument('--skip-ollama-chat', action='store_true')
    parser.add_argument('--skip-ollama-embeddings', action='store_true')
    parser.add_argument('--skip-hf', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    image_bytes = create_mock_chart_png()
    print_torch_status(args.device)
    print_header()
    if not args.skip_ollama_chat:
        for model in OLLAMA_CHAT_MODELS['All-in-one / Reasoning / Code / VLM']:
            print_result(run_ollama_chat(model, image_bytes, args.num_ctx, args.num_predict))
    if not args.skip_ollama_embeddings:
        for model in OLLAMA_EMBEDDING_MODELS:
            print_result(run_ollama_embedding(model))
    if not args.skip_hf:
        for model in HF_IMAGE_TEXT_MODELS:
            print_result(run_hf_image_text(model, image_bytes, args.device, args.hf_dtype))
if __name__ == '__main__':
    main()
