import random
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error

# Keep input format explicit and shared across training/prediction scripts.
DEFAULT_TEXT_COLUMNS: List[str] = [
    "L1_source_word",
    "L1_context",
    "en_target_clue",
    "en_target_word",
]
LABEL_COLUMN = "GLMM_score"
ID_COLUMN = "item_id"

# Local path to pre-downloaded RoBERTa-base model (HPC cluster has no internet)
PRETRAINED_MODEL_PATH = "/gxfs_work/cau/sunpn1134/.cache/huggingface/hub/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_text_columns(df: pd.DataFrame, text_columns: Iterable[str]) -> List[str]:
    cols = [c for c in text_columns if c in df.columns]
    if not cols:
        raise ValueError(
            "No text columns were found in the fold CSV. "
            f"Requested columns: {list(text_columns)}"
        )
    return cols


def build_input_frame(
    df: pd.DataFrame,
    sep_token: str,
    text_columns: Optional[Iterable[str]] = None,
    include_label: bool = True,
) -> pd.DataFrame:
    selected_text_columns = resolve_text_columns(df, text_columns or DEFAULT_TEXT_COLUMNS)

    out = pd.DataFrame()
    out[ID_COLUMN] = df[ID_COLUMN]
    out["input_text"] = (
        df[selected_text_columns]
        .fillna("")
        .astype(str)
        .apply(lambda row: sep_token.join(part.strip() for part in row), axis=1)
    )

    if include_label:
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"Missing required label column: {LABEL_COLUMN}")
        out["label"] = df[LABEL_COLUMN].astype(float)

    return out


def fold_dir(base_dir: Path, outer_fold: int) -> Path:
    return base_dir / f"outer_fold_{outer_fold}"


def model_run_dir(models_dir: Path, outer_fold: int, run_type: str, inner_fold: Optional[int]) -> Path:
    if run_type == "inner":
        if inner_fold is None:
            raise ValueError("inner_fold is required when run_type='inner'")
        return models_dir / f"outer_fold_{outer_fold}" / f"inner_fold_{inner_fold}"

    return models_dir / f"outer_fold_{outer_fold}" / "final"


def rmse_pearson(labels: np.ndarray, preds: np.ndarray) -> dict:
    rmse = root_mean_squared_error(labels, preds)
    corr, _ = pearsonr(preds, labels)
    return {"rmse": float(rmse), "pearson": float(corr)}

