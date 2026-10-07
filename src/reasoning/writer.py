import gc
import json
import os
from pathlib import Path

import peft
import torch
import transformers
from huggingface_hub import HfApi
from transformers.trainer_utils import get_last_checkpoint

from .prompts import chat, encode, parse_response

NAME, REVISION = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# <|im_end|>, <|endoftext|>
STOP_IDS = [248046, 248044]


def train(rows, run_dir, seed, save_every):
    import datasets
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel

    run_dir.mkdir(parents=True, exist_ok=True)
    model, processor = FastVisionModel.from_pretrained(model_name=NAME, revision=REVISION, max_seq_length=8192,
                                                       load_in_4bit=True, use_gradient_checkpointing="unsloth")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    model = FastVisionModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                           finetune_attention_modules=True, finetune_mlp_modules=True,
                                           r=32, lora_alpha=64, lora_dropout=0.0, bias="none", random_state=0)

    dataset = []
    for row in rows:
        x = encode(row, tokenizer)
        if len(x["input_ids"]) <= 8192:
            dataset.append({"input_ids": x["input_ids"], "completion_mask": [int(t != -100) for t in x["labels"]]})
    print(f"train rows {len(dataset)} ({len(rows) - len(dataset)} over 8192 tokens dropped)")

    FastVisionModel.for_training(model)
    args = SFTConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        weight_decay=0.001,
        optim="adamw_8bit",
        logging_steps=5,
        save_steps=save_every,
        save_total_limit=None,
        bf16=True,
        max_length=8192,
        completion_only_loss=True,
        report_to="none",
        seed=seed,
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=datasets.Dataset.from_list(dataset), args=args)
    resume = get_last_checkpoint(str(run_dir))
    result = trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(str(run_dir / "final_adapter"))
    tokenizer.save_pretrained(run_dir / "final_adapter")
    manifest = {"seed": seed, "init_seed": 0, "nf4": True, "rank": 32, "dropout": 0.0, "learning_rate": 2e-4,
                "train_rows": len(dataset), "optimizer_steps": trainer.state.global_step,
                "save_every": save_every, "resumed_from": resume, "train_metrics": result.metrics,
                "gpu": torch.cuda.get_device_name(0), "transformers": transformers.__version__, "peft": peft.__version__}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest)
    push(run_dir)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


def push(run_dir):
    if not Path(".env").exists():
        return
    for line in Path(".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo = os.environ["HF_REPO"]
    api.create_repo(repo, private=True, exist_ok=True)
    path = str(run_dir.relative_to("runs"))
    api.upload_folder(repo_id=repo, folder_path=str(run_dir), path_in_repo=path,
                      ignore_patterns=["*optimizer.pt", "*scheduler.pt", "*rng_state*"])
    print(f"pushed {run_dir} -> {repo}/{path}")
