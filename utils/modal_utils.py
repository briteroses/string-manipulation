import os
import modal
import yaml
from pathlib import Path
from typing import Callable, List

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "modal.yaml"

if modal.is_local():
    with open(CFG_PATH) as cfg_file:
        config = yaml.safe_load(cfg_file)
    model_dir = config["modal"]["model_dir"]
    model_preloads = config["modal"]["model_preloads"]
else:
    model_dir, model_preloads = None, None

def download_all_models_to_modal_image(model_dir: str, model_preloads: str | List[str]):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    if isinstance(model_preloads, str):
        model_preloads = [model_preloads, ]

    for model_name in model_preloads:
        snapshot_download(
            model_name,
            cache_dir=model_dir,
        )
        move_cache()

# llm_inference_image = (
#     modal.Image.debian_slim(python_version="3.10")
#     .pip_install(
#         "accelerate",
#         "anthropic",
#         "base58",
#         "bitsandbytes",
#         "cohere",
#         "datasets",
#         "fastchat",
#         "google-generativeai",
#         "hf-transfer",
#         "huggingface_hub",
#         "openai",
#         "peft",
#         "python-dotenv",
#         "ray",
#         "seaborn",
#         "tiktoken",
#         "tokenizers",
#         "torch",
#         "tqdm",
#         "transformers",
#         "wandb",
#         "vllm",
#     )
#     .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
#     .run_function(
#         download_all_models_to_modal_image,
#         timeout=60 * 20,
#         kwargs={
#             "model_dir": model_dir,
#             "model_preloads": model_preloads,
#         },
#         secrets=[modal.Secret.from_name("PRIVILEGED_HUGGINGFACE_TOKEN")],
#     )
# )

cheap_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "anthropic",
        "base58",
        "bitsandbytes",
        "cohere",
        "datasets",
        "fastchat",
        "google-generativeai",
        "hf-transfer",
        "huggingface_hub",
        "openai",
        "peft",
        "python-dotenv",
        "ray",
        "seaborn",
        "tiktoken",
        "tokenizers",
        "torch",
        "tqdm",
        "transformers",
        "wandb",
        "vllm",
    )
)

app = modal.App("smorgasbord")

vol = modal.Volume.from_name("claude", create_if_missing=True)

config_mounts = [modal.Mount.from_local_dir(Path(__file__).resolve().parents[1] / "configs", remote_path="/root/configs")]

secrets=[
    modal.Secret.from_name(secret_name)
    for secret_name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "WANDB_API_KEY", "ghosts-v-together", "eleven-madison-park")
]

# @app.function(
#     image=llm_inference_image,
#     gpu=modal.gpu.A100(size="40GB", count=1),
#     secrets=[modal.Secret.from_name("OPENAI_API_KEY"), modal.Secret.from_name("ANTHROPIC_API_KEY"), modal.Secret.from_name("WANDB_API_KEY"), modal.Secret.from_name("PRIVILEGED_HUGGINGFACE_TOKEN")],
#     mounts=config_mounts,
#     volumes={"/data": vol},
#     _allow_background_volume_commits=True,
#     timeout=86400, # seconds in 1 day, the maximum allowed time
# )
# def modal_wrap_experiment(runner: Callable, *runner_args, **runner_kwargs):
#     return runner(*runner_args, **runner_kwargs)

@app.function(
    image=cheap_image,
    secrets=secrets,
    mounts=config_mounts,
    volumes={"/data": vol},
    _allow_background_volume_commits=True,
    timeout=86400, # seconds in 1 day, the maximum allowed time
)
def cheap_modal_wrap_experiment(runner: Callable, *runner_args, **runner_kwargs):
    return runner(*runner_args, **runner_kwargs)