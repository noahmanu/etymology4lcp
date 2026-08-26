#!/usr/bin/env python3
"""Create CV predictions with trained RoBERTa-base fold models."""

import argparse
import json
from pathlib import Path

from datasets import Dataset
import numpy as np
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
    ID_COLUMN,
    build_input_frame,
    fold_dir,
    model_run_dir,
    rmse_pearson,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict GLMM scores with trained RoBERTa CV models.")
    parser.add_argument("--folds-dir", type=Path, default=Path("../data/feature_enriched/folds"))
    parser.add_argument("--models-dir", type=Path, default=Path("./roberta_models"))
    parser.add_argument("--output-dir", type=Path, default=Path("./roberta_predictions"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def build_tokenized_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    sep_token = f" {tokenizer.sep_token} " if tokenizer.sep_token else " "
    input_df = build_input_frame(df, sep_token, text_columns=DEFAULT_TEXT_COLUMNS, include_label=True)
    dataset = Dataset.from_pandas(input_df[["input_text", "label"]], preserve_index=False)
    return dataset.map(
        lambda batch: tokenizer(batch["input_text"], truncation=True, max_length=max_length),
        batched=True,
        desc="Tokenizing",
    )


def predict_with_model(model_dir: Path, tokenized_ds: Dataset, batch_size: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorWithPadding(tokenizer),
        args=TrainingArguments(output_dir=str(model_dir / "predict_tmp"), report_to="none", per_device_eval_batch_size=batch_size),
    )

    predictions = trainer.predict(tokenized_ds).predictions.flatten()
    return predictions


def save_predictions(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(ID_COLUMN).to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()

    for outer_fold in [1, 2, 3, 4, 5]:
        outer_dir = fold_dir(args.folds_dir, outer_fold)
        outer_train_df = pd.read_csv(outer_dir / "outer_train.csv")
        oof_frames = []

        # Build OOF predictions for outer_train from the 4 inner models.
        for inner_fold in [1, 2, 3, 4]:
            model_dir = model_run_dir(args.models_dir, outer_fold, "inner", inner_fold)
            inner_test_df = pd.read_csv(outer_dir / f"inner_fold_{inner_fold}" / "inner_test.csv")
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
            tokenized_ds = build_tokenized_dataset(inner_test_df, tokenizer, args.max_length)
            preds = predict_with_model(model_dir, tokenized_ds, args.batch_size)

            fold_pred_df = pd.DataFrame(
                {
                    ID_COLUMN: inner_test_df[ID_COLUMN].values,
                    "roberta_base_score": preds,
                    "GLMM_score": inner_test_df["GLMM_score"].values,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                }
            )
            oof_frames.append(fold_pred_df)

        oof_df = pd.concat(oof_frames, ignore_index=True)
        if oof_df[ID_COLUMN].nunique() != outer_train_df[ID_COLUMN].nunique():
            raise ValueError(f"OOF coverage mismatch in outer fold {outer_fold}")

        metrics = rmse_pearson(oof_df["GLMM_score"].to_numpy(), oof_df["roberta_base_score"].to_numpy())

        outer_output_dir = args.output_dir / f"outer_fold_{outer_fold}"
        save_predictions(oof_df[[ID_COLUMN, "roberta_base_score", "GLMM_score", "inner_fold"]], outer_output_dir / "outer_train_oof_predictions.csv")

        # Predict outer test with final model trained on full outer train.
        final_model_dir = model_run_dir(args.models_dir, outer_fold, "final", None)
        outer_test_df = pd.read_csv(outer_dir / "outer_test.csv")
        tokenizer = AutoTokenizer.from_pretrained(final_model_dir, use_fast=True, local_files_only=True)
        tokenized_test_ds = build_tokenized_dataset(outer_test_df, tokenizer, args.max_length)
        outer_test_preds = predict_with_model(final_model_dir, tokenized_test_ds, args.batch_size)

        outer_test_pred_df = pd.DataFrame(
            {
                ID_COLUMN: outer_test_df[ID_COLUMN].values,
                "roberta_base_score": outer_test_preds,
                "GLMM_score": outer_test_df["GLMM_score"].values,
                "outer_fold": outer_fold,
            }
        )
        save_predictions(outer_test_pred_df, outer_output_dir / "outer_test_predictions.csv")

        (outer_output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"outer_fold_{outer_fold}: OOF RMSE={metrics['rmse']:.4f}, Pearson={metrics['pearson']:.4f}")


if __name__ == "__main__":
    main()



