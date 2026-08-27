#!/usr/bin/env python3
"""Train one RoBERTa-base model for a specific outer/inner CV run."""

import argparse
import json
from pathlib import Path

from datasets import Dataset
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from roberta_cv_utils import (
    DEFAULT_TEXT_COLUMNS,
    PRETRAINED_MODEL_PATH,
    build_input_frame,
    fold_dir,
    model_run_dir,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa-base for one CV run.")
    parser.add_argument("--outer-fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--run-type", type=str, required=True, choices=["inner", "final"])
    parser.add_argument("--inner-fold", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--folds-dir", type=Path, default=Path("../data/feature_enriched/folds"))
    parser.add_argument("--models-dir", type=Path, default=Path("./roberta_models"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=667)
    return parser.parse_args()


def load_split_csv(base_fold_dir: Path, run_type: str, inner_fold: int | None) -> Path:
    if run_type == "inner":
        if inner_fold is None:
            raise ValueError("--inner-fold is required when --run-type inner")
        return base_fold_dir / f"inner_fold_{inner_fold}" / "inner_train.csv"
    return base_fold_dir / "outer_train.csv"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    base_fold_dir = fold_dir(args.folds_dir, args.outer_fold)
    train_csv = load_split_csv(base_fold_dir, args.run_type, args.inner_fold)
    run_dir = model_run_dir(args.models_dir, args.outer_fold, args.run_type, args.inner_fold)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, use_fast=True, local_files_only=True)
    sep_token = f" {tokenizer.sep_token} " if tokenizer.sep_token else " "

    train_df = pd.read_csv(train_csv)
    input_df = build_input_frame(train_df, sep_token, text_columns=DEFAULT_TEXT_COLUMNS, include_label=True)

    train_ds = Dataset.from_pandas(input_df[["input_text", "label"]], preserve_index=False)
    tokenized_train = train_ds.map(
        lambda batch: tokenizer(batch["input_text"], truncation=True, max_length=args.max_length),
        batched=True,
        desc="Tokenizing train data",
    )

    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH, num_labels=1, local_files_only=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=3e-5,
        weight_decay=0.1,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        seed=args.seed,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)

    metadata = {
        "outer_fold": args.outer_fold,
        "run_type": args.run_type,
        "inner_fold": args.inner_fold,
        "train_csv": str(train_csv),
        "num_rows": int(len(train_df)),
        "pretrained_model": PRETRAINED_MODEL_PATH,
        "learning_rate": 3e-5,
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()



