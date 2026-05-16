from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LoRA governor adapter on Qwen 0.5B-Instruct.")
    p.add_argument("--config", default="configs/governor.yaml")
    p.add_argument("--dataset-jsonl", default="data/processed/research/lora_governor.jsonl")
    p.add_argument("--base", default=None, help="Override base model id (e.g. Qwen/Qwen2.5-Coder-1.5B for fallback)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    lt = cfg["lora_training"]
    base_dir = args.base or lt["base_model_dir"]
    out_root = Path(lt["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Loading base {base_dir} for LoRA training")
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16).to(device)
    peft_cfg = LoraConfig(
        r=int(lt["rank"]),
        lora_alpha=int(lt["alpha"]),
        target_modules=list(lt["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_cfg)

    ds = load_dataset("json", data_files=args.dataset_jsonl, split="train")
    held_out_n = max(1, int(len(ds) * float(lt["held_out_fraction"])))
    eval_ds = ds.select(range(held_out_n))
    train_ds = ds.select(range(held_out_n, len(ds)))

    def render(example):
        text = ""
        for m in example["messages"]:
            text += f"<|im_start|>{m['role']}\n{m['content']}\n<|im_end|>\n"
        toks = tok(text, truncation=True, max_length=int(lt["max_seq_length"]), padding="max_length")
        toks["labels"] = toks["input_ids"]
        return toks

    train_tok = train_ds.map(render, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(render, remove_columns=eval_ds.column_names)

    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=int(lt["max_epochs"]),
        per_device_train_batch_size=int(lt["batch_size"]),
        gradient_accumulation_steps=int(lt["gradient_accumulation_steps"]),
        learning_rate=float(lt["learning_rate"]),
        warmup_steps=int(lt["warmup_steps"]),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=int(lt["random_seed"]),
        bf16=device == "mps",
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_tok, eval_dataset=eval_tok)
    started = time.time()
    trainer.train()
    elapsed_h = (time.time() - started) / 3600.0

    eval_metrics = trainer.evaluate()
    held_out_perplexity = float(eval_metrics.get("eval_loss", 0.0))

    model.save_pretrained(run_dir)
    tok.save_pretrained(run_dir)

    metrics = {
        "run_id": run_id,
        "base_model_dir": base_dir,
        "elapsed_hours": elapsed_h,
        "held_out_perplexity": held_out_perplexity,
        "n_train_records": len(train_ds),
        "n_eval_records": len(eval_ds),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    latest = out_root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_id)

    console.print(f"Trained adapter at {run_dir}; metrics.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
