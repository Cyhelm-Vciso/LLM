"""LoRA / QLoRA supervised fine-tuning with Hugging Face TRL.

Heavy dependencies are imported lazily so the rest of the toolkit works without a GPU.
Install with: pip install -e '.[train]'
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("base_model", "train_file", "output_dir"):
        if key not in cfg:
            raise SystemExit(f"{path}: missing required key '{key}'")
    if not Path(cfg["train_file"]).exists():
        raise SystemExit(f"train_file not found: {cfg['train_file']} (run `cyberllm build` first)")
    return cfg


def train(cfg: dict) -> str:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:  # pragma: no cover
        raise SystemExit(f"Training dependencies missing ({e.name}). Run: pip install -e '.[train]'") from e

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    quant = cfg.get("quantization", "4bit")
    bnb = None
    if quant == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif quant == "8bit":
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit(f"{cfg['base_model']} has no chat template; pick an instruct/chat model.")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        torch_dtype=dtype,
        device_map="auto",
    )

    data_files = {"train": cfg["train_file"]}
    if cfg.get("eval_file") and Path(cfg["eval_file"]).exists() and Path(cfg["eval_file"]).stat().st_size > 0:
        data_files["eval"] = cfg["eval_file"]
    ds = load_dataset("json", data_files=data_files)

    lora = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=lora.get("target_modules", "all-linear"),
        task_type="CAUSAL_LM",
    )

    has_eval = "eval" in ds
    args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 3),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        eval_strategy="steps" if has_eval else "no",
        eval_steps=cfg.get("eval_steps", 100) if has_eval else None,
        max_length=cfg.get("max_length", 4096),
        assistant_only_loss=cfg.get("assistant_only_loss", False),
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        seed=cfg.get("seed", 42),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"] if has_eval else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    return cfg["output_dir"]


def merge(adapter_dir: str, out_dir: str) -> str:
    """Merge a LoRA adapter into its base model for serving with vLLM/Ollama/llama.cpp."""
    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise SystemExit(f"Training dependencies missing ({e.name}). Run: pip install -e '.[train]'") from e
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, torch_dtype=torch.bfloat16, device_map="auto")
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
    return out_dir
