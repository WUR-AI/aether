"""
Interactive terminal UI to guide users through the Crop Yield Africa
preprocessing pipeline in the correct order.

Usage:
    python src/data_preprocessing/yield_africa_pipeline_tui.py

Optional enhancement (better rendering):
    pip install rich        # or: uv add rich --dev

No extra packages required — falls back to a plain-ANSI terminal UI.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Load .env early — before any os.environ reads
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load the project .env file into os.environ.

    Tries python-dotenv first (already a project dependency via train.py).
    Falls back to a minimal built-in parser that handles the subset of syntax
    used in this project's .env.example:
      KEY="value"           quoted or unquoted values
      KEY="${OTHER}/sub"    simple ${VAR} interpolation (already-resolved vars)
      # comment lines       skipped
      blank lines           skipped
    """
    # Locate the .env file: walk up from this script's directory until we find
    # a .project-root marker or the filesystem root.
    start = Path(__file__).resolve().parent
    root = start
    for parent in [start, *start.parents]:
        if (parent / ".project-root").exists() or (parent / ".env").exists():
            root = parent
            break

    env_file = root / ".env"
    if not env_file.exists():
        return

    # Prefer python-dotenv (available in the project's venv).
    try:
        from dotenv import load_dotenv as _ld
        _ld(dotenv_path=env_file, override=False)
        return
    except ImportError:
        pass

    # Minimal fallback parser.
    import re
    resolved: dict[str, str] = {}
    with env_file.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # Expand ${VAR} references using already-resolved values then os.environ.
            def _expand(m: re.Match) -> str:
                name = m.group(1)
                return resolved.get(name, os.environ.get(name, m.group(0)))
            val = re.sub(r"\$\{([^}]+)\}", _expand, val)
            resolved[key] = val
            if key not in os.environ:
                os.environ[key] = val


_load_dotenv()

# ---------------------------------------------------------------------------
# Optional rich import
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.prompt import Prompt, Confirm
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.markup import escape
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ---------------------------------------------------------------------------
# ANSI colour helpers (used when rich is not available)
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

FG_BLACK   = "\033[30m"
FG_RED     = "\033[31m"
FG_GREEN   = "\033[32m"
FG_YELLOW  = "\033[33m"
FG_BLUE    = "\033[34m"
FG_MAGENTA = "\033[35m"
FG_CYAN    = "\033[36m"
FG_WHITE   = "\033[37m"
FG_BRIGHT_WHITE = "\033[97m"

BG_BLUE    = "\033[44m"
BG_CYAN    = "\033[46m"
BG_GREEN   = "\033[42m"

def _c(*parts: str) -> str:
    return "".join(parts)

def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PipelineStep:
    id: str
    title: str
    description: str
    script: str
    required: bool
    depends_on: List[str] = field(default_factory=list)
    args_hint: str = ""
    output_hint: str = ""
    extra_deps: List[str] = field(default_factory=list)
    advice: str = ""

STEPS: List[PipelineStep] = [
    PipelineStep(
        id="ai_readiness",
        title="0. AI-readiness analysis (raw data)",
        description=(
            "Exploratory analysis of the RAW input CSV before any preprocessing.\n"
            "Produces a data-quality report (missingness, target distribution,\n"
            "country/year balance, location accuracy), a Random Forest benchmark\n"
            "with spatial group CV, permutation importance, PCA redundancy analysis,\n"
            "residual maps, a feature correlation heatmap, and a structured\n"
            "AI-readiness gap report with strengths, gaps, and next steps."
        ),
        script="src/data_preprocessing/yield_africa_ai_readiness_analysis.py",
        required=False,
        depends_on=[],
        args_hint=(
            "--input_csv data/yield_africa/Full_dataset_CropYield_classified_and_numeric.csv\n"
            "[--out_dir  data/yield_africa/analysis_ai_readiness]\n"
            "[--model    both]\n"
            "[--n_trees  300]\n"
            "[--xgb_n_estimators 300]\n"
            "[--seed     42]"
        ),
        output_hint=(
            "data/yield_africa/analysis_ai_readiness/feature_importance.png\n"
            "data/yield_africa/analysis_ai_readiness/pca_scree.png\n"
            "data/yield_africa/analysis_ai_readiness/residuals.png\n"
            "data/yield_africa/analysis_ai_readiness/feature_correlation.png\n"
            "data/yield_africa/analysis_ai_readiness/xgb/feature_importance.png\n"
            "data/yield_africa/analysis_ai_readiness/xgb/residuals.png"
        ),
        advice=(
            "Requires the RICH semicolon-delimited raw CSV with soil, climate,\n"
            "terrain, and location-accuracy columns (the 'classified_and_numeric'\n"
            "version, NOT yield_africa_original.csv which has only 10 columns).\n"
            "Run once before step 1 to assess data quality and gaps.\n"
            "Use --model rf for Random Forest only, --model xgb for XGBoost only,\n"
            "or --model both (default) to run both. XGBoost outputs go to out_dir/xgb/.\n"
            "The AI-readiness report printed to stdout summarises strengths, gaps,\n"
            "and recommended next steps for model development."
        ),
    ),
    PipelineStep(
        id="make_ready",
        title="1. Build model-ready CSV",
        description=(
            "Transforms raw yield data into a clean, feature-engineered CSV.\n"
            "Computes derived features (CN ratio, WHC proxy, aridity index),\n"
            "removes yield outliers, log-transforms skewed columns, and fits\n"
            "label encoders on training countries only to prevent leakage."
        ),
        script="src/data_preprocessing/make_model_ready_yield_africa.py",
        required=True,
        args_hint=(
            "--source_csv data/yield_africa/yield_africa_v20260218.csv\n"
            "--out_csv    data/yield_africa/model_ready_yield_africa.csv\n"
            "[--spatial_splits]   # optional: 7-fold 50 km block CV splits"
        ),
        output_hint=(
            "data/yield_africa/model_ready_yield_africa.csv\n"
            "data/yield_africa/label_encoders_yield_africa.pkl\n"
            "[data/yield_africa/splits/split_spatial_7_folds_*.pth]"
        ),
        advice=(
            "This step must be run before any augmentation or split generation.\n"
            "The label encoders are fitted on TRAIN_COUNTRIES only; keep this\n"
            "file safe — you will need it at inference time."
        ),
    ),
    PipelineStep(
        id="augment_ndvi",
        title="2a. Augment with NDVI",
        description=(
            "Attaches time-variant, year-specific NDVI features to each record\n"
            "by spatially matching each plot to pre-computed monthly NDVI grids.\n"
            "Adds 12 monthly features (feat_ndvi_month_1…12) and 5 seasonal\n"
            "aggregates (MAM, JJA, SON, DJF, growing season)."
        ),
        script="src/data_preprocessing/yield_africa_augment_ndvi.py",
        required=False,
        depends_on=["make_ready"],
        args_hint=(
            "--input_csv  data/yield_africa/model_ready_yield_africa.csv\n"
            "--output_csv data/yield_africa/model_ready_yield_africa_ndvi.csv\n"
            "--ndvi_dir   data/cache/ndvi\n"
            "[--download]                    # fetch missing years from NASA MODIS\n"
            "[--modis_product MOD13A3]       # MOD13A3 (monthly 1km, ~1.3 GB/yr, default)\n"
            "                                # or MOD13Q1 (16-day 250m, ~44 GB/yr)\n"
            "[--workers 4]                   # parallel threads for HDF processing"
        ),
        output_hint="data/yield_africa/model_ready_yield_africa_ndvi.csv",
        extra_deps=["earthaccess  (only if using --download flag)"],
        advice=(
            "Pre-computed NDVI_100m_{year}.csv files must be present in ndvi_dir\n"
            "unless you use --download. Set EARTHDATA_USERNAME / EARTHDATA_PASSWORD\n"
            "in .env when downloading from NASA EarthData.\n"
            "Use --modis_product MOD13A3 (default) for ~34× smaller downloads vs MOD13Q1."
        ),
    ),
    PipelineStep(
        id="augment_agera5",
        title="2b. Augment with AgERA5 climate",
        description=(
            "Fetches year-specific daily climate data (temperature, precipitation,\n"
            "radiation, wind, vapour pressure) from the AgERA5/CHIRPS web API for\n"
            "every unique (lat, lon, year) triplet and computes seasonal aggregates\n"
            "plus growing-degree-days and wet-day counts."
        ),
        script="src/data_preprocessing/yield_africa_augment_agera5.py",
        required=False,
        depends_on=["make_ready"],
        args_hint=(
            "--input_csv  data/yield_africa/model_ready_yield_africa.csv\n"
            "--output_csv data/yield_africa/model_ready_yield_africa_agera5.csv\n"
            "--cache_dir  data/cache/agera5\n"
            "--workers    4\n"
            "--timeout    120"
        ),
        output_hint=(
            "data/yield_africa/model_ready_yield_africa_agera5.csv\n"
            "data/cache/agera5/agera5_*.json  (resumable cache)"
        ),
        advice=(
            "This step makes live HTTP requests and can be slow (~minutes to hours).\n"
            "It is fully resumable — raw API responses are cached as JSON files, so\n"
            "you can safely interrupt and re-run. Use more --workers for speed."
        ),
    ),
    PipelineStep(
        id="tessera",
        title="2c. Download TESSERA embeddings",
        description=(
            "Downloads per-record, year-specific satellite embeddings from the\n"
            "GeoTessera service. Each record gets a small EO tile (default 9 px)\n"
            "that captures local land-cover and phenology at the plot location."
        ),
        script="src/data_preprocessing/yield_africa_tessera_preprocess.py",
        required=False,
        depends_on=["make_ready"],
        args_hint=(
            "--data_dir  data/\n"
            "[--countries KEN RWA]   # subset of countries\n"
            "[--tile_size 9]         # pixels around plot centre"
        ),
        output_hint=(
            "data/yield_africa/eo/tessera/tessera_{name_loc}.npy\n"
            "(one NumPy array per plot)"
        ),
        extra_deps=["geotessera  (install with: uv sync --extra geotessera)"],
        advice=(
            "Install the optional dependency first:\n"
            "  uv sync --extra geotessera\n"
            "Set TESSERA_EMBEDDINGS_DIR in .env to point to a directory with\n"
            "enough disk space (can be on an external drive).\n"
            "This step is resumable — existing .npy files are skipped."
        ),
    ),
    PipelineStep(
        id="merge_augmentations",
        title="2d. Merge augmented CSVs",
        description=(
            "Merges the base model-ready CSV with any combination of NDVI and\n"
            "AgERA5 augmented CSVs into a single combined CSV.  Shared columns\n"
            "(all base feat_* columns, join keys, target and aux columns) appear\n"
            "exactly once — no duplication.  The AgERA5 sentinel column\n"
            "'agera5_fetched' is preserved.  Use --how left (default) to keep all\n"
            "base rows and fill missing augmentation values with NaN, or --how\n"
            "inner to keep only rows present in every provided CSV."
        ),
        script="src/data_preprocessing/yield_africa_merge_augmentations.py",
        required=False,
        depends_on=["make_ready"],
        args_hint=(
            "--base_csv   data/yield_africa/model_ready_yield_africa.csv\n"
            "[--ndvi_csv   data/yield_africa/model_ready_yield_africa_ndvi.csv]\n"
            "[--agera5_csv data/yield_africa/model_ready_yield_africa_agera5.csv]\n"
            "[--out_csv    data/yield_africa/model_ready_yield_africa_merged.csv]\n"
            "[--how        left]"
        ),
        output_hint=(
            "data/yield_africa/model_ready_yield_africa_merged.csv"
        ),
        advice=(
            "Run after whichever augmentation steps (2a, 2b) you have completed.\n"
            "At least one of --ndvi_csv or --agera5_csv must be provided.\n"
            "The merged CSV can be used directly in any experiment config:\n"
            "  data.dataset.csv_name: model_ready_yield_africa_merged.csv\n"
            "Use --how inner if you only want rows with complete augmentation data;\n"
            "use --how left (default) to retain all base rows with NaN for missing\n"
            "augmentation rows (the model will impute them at runtime)."
        ),
    ),
    PipelineStep(
        id="check_augmentations",
        title="2e. Compare all augmentations",
        description=(
            "Benchmarks any combination of augmented CSVs (base, +NDVI, +AgERA5)\n"
            "side by side: RF performance, feature importance share per augmentation,\n"
            "PCA dimensionality, and per-country holdout R². Only --base_csv is\n"
            "required — omit --ndvi_csv or --agera5_csv to skip those comparisons.\n"
            "Produces a ranked recommendation and Hydra config snippets."
        ),
        script="src/data_preprocessing/yield_africa_augmentation_comparison.py",
        required=False,
        depends_on=["make_ready"],
        args_hint=(
            "--base_csv   data/yield_africa/model_ready_yield_africa.csv\n"
            "[--ndvi_csv   data/yield_africa/model_ready_yield_africa_ndvi.csv]\n"
            "[--agera5_csv data/yield_africa/model_ready_yield_africa_agera5.csv]\n"
            "[--merged_csv data/yield_africa/model_ready_yield_africa_merged.csv]\n"
            "--out_dir    data/yield_africa/analysis_augmentation\n"
            "[--model     both]\n"
            "[--n_trees   300]\n"
            "[--xgb_n_estimators 300]\n"
            "[--seed      42]\n"
            "[--complete_only]"
        ),
        output_hint=(
            "data/yield_africa/analysis_augmentation/rf_comparison.png\n"
            "data/yield_africa/analysis_augmentation/xgb_comparison.png\n"
            "data/yield_africa/analysis_augmentation/importance_*.png\n"
            "data/yield_africa/analysis_augmentation/xgb/importance_*.png\n"
            "data/yield_africa/analysis_augmentation/pca_comparison.png\n"
            "data/yield_africa/analysis_augmentation/per_country_r2.png"
        ),
        advice=(
            "Run after step 1, and after whichever augmentation steps (2a, 2b) you\n"
            "have completed — only the CSVs you provide are compared.\n"
            "Use --model rf to run Random Forest only, --model xgb for XGBoost only,\n"
            "or --model both (default) to run both and compare results side by side.\n"
            "XGBoost importance and per-country plots are saved under out_dir/xgb/.\n"
            "Use --complete_only to restrict the comparison to rows that actually\n"
            "have augmentation data (no NaN imputation), which is useful when downloads\n"
            "are still in progress and many rows are incomplete."
        ),
    ),
    PipelineStep(
        id="spatial_splits",
        title="3a. Generate spatial splits",
        description=(
            "Creates geographically-aware train/val/test splits using DBSCAN\n"
            "clustering so that no nearby plots straddle a split boundary.\n"
            "Prevents spatial data leakage during cross-validation."
        ),
        script="src/data_preprocessing/yield_africa_spatial_splits.py",
        required=False,
        depends_on=["make_ready"],
        args_hint=(
            "--data_dir      data/\n"
            "[--distance_km  10 25 50]   # cluster radii to generate"
        ),
        output_hint=(
            "data/yield_africa/splits/split_spatial_10km.pth\n"
            "data/yield_africa/splits/split_spatial_25km.pth\n"
            "data/yield_africa/splits/split_spatial_50km.pth"
        ),
        advice=(
            "Generate multiple distances to compare how spatial autocorrelation\n"
            "affects model generalisation. The 50 km split is used in the default\n"
            "yield_africa_spatial.yaml data config."
        ),
    ),
    PipelineStep(
        id="loco_splits",
        title="3b. Generate LOCO splits",
        description=(
            "Creates leave-one-country-out splits for evaluating how well the\n"
            "model generalises to unseen countries. One .pth file per country\n"
            "holds out that entire country as the test set."
        ),
        script="src/data_preprocessing/yield_africa_loco_splits.py",
        required=False,
        depends_on=["make_ready"],
        args_hint=(
            "--data_dir  data/\n"
            "[--country  KEN]   # generate for a single country only"
        ),
        output_hint=(
            "data/yield_africa/splits/split_loco_BUR.pth\n"
            "data/yield_africa/splits/split_loco_ETH.pth\n"
            "data/yield_africa/splits/split_loco_KEN.pth  … (all 8 countries)"
        ),
        advice=(
            "The default data config yield_africa_loco.yaml holds out Tanzania (TAN).\n"
            "To test another country, point data.saved_split_file_name at the\n"
            "appropriate .pth file in your experiment config."
        ),
    ),
]

STEP_INDEX = {s.id: s for s in STEPS}

EXPERIMENTS = [
    ("yield_africa_coords_reg",           "Coordinates only (baseline)"),
    ("yield_africa_tabular_reg",          "Tabular soil/climate features"),
    ("yield_africa_fusion_reg",           "Tabular + coordinate fusion"),
    ("yield_africa_tabular_spatial",      "Tabular + spatial split"),
    ("yield_africa_fusion_spatial",       "Fusion + spatial split"),
    ("yield_africa_tabular_loco",         "Tabular + LOCO split"),
    ("yield_africa_fusion_loco",          "Fusion + LOCO split"),
    ("yield_africa_tessera_reg",          "TESSERA embeddings"),
    ("yield_africa_tessera_fusion_reg",   "TESSERA + tabular fusion"),
    ("yield_africa_tessera_fusion_spatial","TESSERA fusion + spatial split"),
    ("yield_africa_tessera_fusion_loco",  "TESSERA fusion + LOCO split"),
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns

def _data_dir() -> Path:
    """Return the resolved DATA_DIR path (never a bare 'data/' relative path)."""
    raw = os.environ.get("DATA_DIR", "").strip().rstrip("/")
    if raw:
        return Path(raw)
    # Fall back: DATA_DIR defaults to <PROJECT_ROOT>/data/
    project_root = os.environ.get("PROJECT_ROOT", "").strip().rstrip("/")
    if project_root:
        return Path(project_root) / "data"
    # Last resort: resolve relative to the project root marker
    start = Path(__file__).resolve().parent
    for parent in [start, *start.parents]:
        if (parent / ".project-root").exists():
            return parent / "data"
    return Path("data")

def _expand_paths(text: str) -> str:
    """Replace the 'data/' prefix in path tokens with the resolved DATA_DIR."""
    data = str(_data_dir())
    import re
    # Replace bare 'data/' at the start of a path token (word boundary or line start)
    return re.sub(r"(?<![/\w])data/", data + "/", text)

def _check_env() -> dict:
    """Return a dict of env-var status."""
    keys = [
        "PROJECT_ROOT",
        "DATA_DIR",
        "TRAINER_PROFILE",
        "TESSERA_EMBEDDINGS_DIR",
        "EARTHDATA_USERNAME",
        "EARTHDATA_PASSWORD",
    ]
    return {k: os.environ.get(k) for k in keys}

def _check_output_exists(step: PipelineStep) -> Optional[bool]:
    """Best-effort check whether the step's primary output exists."""
    first_line = step.output_hint.split("\n")[0].strip().split("(")[0].strip()
    if not first_line or first_line.startswith("["):
        return None
    return Path(_expand_paths(first_line)).exists()

def _script_path(step: PipelineStep) -> Path:
    """Resolve script path relative to project root (best effort)."""
    root = os.environ.get("PROJECT_ROOT", ".")
    return Path(root) / step.script

@dataclass
class _OptionalArg:
    """One optional argument parsed from an args_hint line."""
    raw: str          # full token string, e.g. "--ndvi_csv /abs/path/foo.csv"
    flag: str         # just the flag name, e.g. "--ndvi_csv"
    value: str        # the value portion, e.g. "/abs/path/foo.csv"  (empty for bare flags)
    is_path: bool     # True when value looks like an absolute or data-relative path


def _parse_optional_args(step: PipelineStep) -> tuple[str, list[_OptionalArg]]:
    """Parse args_hint and return (required_cmd, [_OptionalArg, ...]).

    Lines wrapped in [...] are optional; all others are required.
    Comment tokens (#...) are stripped. Paths are expanded via _expand_paths.
    """
    import re
    required_parts: list[str] = []
    optional_args: list[_OptionalArg] = []

    for raw_line in _expand_paths(step.args_hint).splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            continue
        line = re.sub(r"\s+#.*$", "", line).strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            inner = line[1:-1].strip()
            # Split flag from value: "--flag value" or just "--flag"
            parts = inner.split(None, 1)
            flag = parts[0]
            value = parts[1] if len(parts) > 1 else ""
            # It's a path if the value looks like an absolute path or contains /
            is_path = bool(value) and ("/" in value or value.startswith("."))
            optional_args.append(_OptionalArg(raw=inner, flag=flag, value=value, is_path=is_path))
        else:
            required_parts.append(line)

    cmd = "uv run python " + step.script
    if required_parts:
        cmd += " " + " ".join(required_parts)
    return cmd, optional_args


def _prompt_optional_args_plain(optional_args: list[_OptionalArg]) -> str:
    """Interactively prompt for each optional arg in plain-ANSI mode.
    Returns a string of extra args to append to the command (may be empty).
    """
    if not optional_args:
        return ""
    extras: list[str] = []
    print()
    print(BOLD + FG_YELLOW + "  Optional arguments:" + RESET)
    for opt in optional_args:
        if opt.is_path:
            exists_hint = FG_GREEN + " (file exists)" + RESET if Path(opt.value).exists() else DIM + " (not found)" + RESET
            prompt_str = (
                f"\n  Include {BOLD}{opt.flag}{RESET}?\n"
                f"  {DIM}{opt.value}{RESET}{exists_hint}\n"
                f"  [y/N] "
            )
            ans = input(prompt_str).strip().lower()
            if ans == "y":
                extras.append(f"{opt.flag} {opt.value}")
        else:
            if opt.value:
                prompt_str = (
                    f"\n  {BOLD}{opt.flag}{RESET} "
                    f"[default: {FG_CYAN}{opt.value}{RESET}, Enter to keep, or type new value, or 's' to skip]: "
                )
                ans = input(prompt_str).strip()
                if ans.lower() == "s":
                    pass
                elif ans == "":
                    extras.append(f"{opt.flag} {opt.value}")
                else:
                    extras.append(f"{opt.flag} {ans}")
            else:
                # Bare flag (e.g. --spatial_splits, --download)
                prompt_str = f"\n  Include flag {BOLD}{opt.flag}{RESET}? [y/N] "
                ans = input(prompt_str).strip().lower()
                if ans == "y":
                    extras.append(opt.flag)
    return (" " + " ".join(extras)) if extras else ""

# ===========================================================================
# RICH UI  (when rich is available)
# ===========================================================================

def run_rich() -> None:
    console = Console()

    def header() -> None:
        console.print()
        console.print(Panel(
            Text.from_markup(
                "[bold bright_green]AETHER-xAI[/] [cyan]|[/] "
                "[bold white]Crop Yield Africa[/] [cyan]|[/] "
                "[bold yellow]Preprocessing Pipeline[/]\n"
                "[dim]Interactive guide to prepare data for model training[/]"
            ),
            box=box.DOUBLE_EDGE,
            border_style="cyan",
            padding=(0, 2),
        ))

    def _step_status_icon(step: PipelineStep) -> Text:
        exists = _check_output_exists(step)
        if step.required:
            if exists:
                return Text("✓ done", style="bold green")
            return Text("● required", style="bold yellow")
        if exists:
            return Text("✓ done", style="green")
        return Text("○ optional", style="dim")

    def show_overview() -> None:
        header()
        console.print(Rule("[bold cyan]Pipeline Overview[/]", style="cyan"))
        console.print()

        table = Table(
            box=box.ROUNDED,
            border_style="blue",
            show_header=True,
            header_style="bold cyan",
            expand=False,
            padding=(0, 1),
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Step", style="bold white", min_width=28)
        table.add_column("Required", justify="center", width=10)
        table.add_column("Status", width=12)
        table.add_column("Script", style="dim", min_width=30)

        for step in STEPS:
            req = Text("YES", style="bold yellow") if step.required else Text("no", style="dim")
            table.add_row(
                step.id.split("_")[0][:3],
                step.title,
                req,
                _step_status_icon(step),
                step.script,
            )

        console.print(table)
        console.print()
        console.print(
            "[dim]Step 2a/2b/2c are independent augmentation steps that can be run\n"
            "in any order after step 1. Step 2d merges augmented CSVs; run after\n"
            "2a/2b as needed. Steps 3a/3b generate split files and\n"
            "also only need step 1 to be completed first.[/]"
        )

    def show_env() -> None:
        header()
        console.print(Rule("[bold cyan]Environment Variables[/]", style="cyan"))
        console.print()

        env = _check_env()
        table = Table(box=box.SIMPLE, border_style="dim", header_style="bold cyan", expand=False)
        table.add_column("Variable", style="bold")
        table.add_column("Value / Status")

        for k, v in env.items():
            if v:
                table.add_row(k, Text(v, style="green"))
            else:
                style = "red" if k in ("PROJECT_ROOT", "DATA_DIR") else "dim"
                table.add_row(k, Text("not set", style=style))

        console.print(table)
        console.print()

        required_missing = [k for k in ("PROJECT_ROOT", "DATA_DIR") if not env.get(k)]
        if required_missing:
            console.print(Panel(
                "[bold red]WARNING:[/] Required variables not set: "
                + ", ".join(required_missing)
                + "\nEdit your [bold].env[/] file — this TUI loads it automatically on startup.",
                border_style="red",
                padding=(0, 1),
            ))
        else:
            console.print("[green]All required env vars are set.[/]")

    def show_step_detail(step: PipelineStep) -> None:
        header()
        console.print(Rule(f"[bold cyan]{step.title}[/]", style="cyan"))
        console.print()

        # Description
        console.print(Panel(
            step.description,
            title="[bold]What it does[/]",
            border_style="blue",
            padding=(0, 1),
        ))

        # Dependency
        if step.depends_on:
            deps = ", ".join(step.depends_on)
            console.print(f"\n[yellow]Depends on:[/] {deps}")

        # Extra deps
        if step.extra_deps:
            console.print()
            console.print("[bold yellow]Optional extra packages:[/]")
            for d in step.extra_deps:
                console.print(f"  [dim]•[/] {d}")

        # Args hint
        console.print()
        console.print(Panel(
            f"[bold green]uv run python {step.script}[/]\n"
            + "\n".join(
                f"  [cyan]{escape(_expand_paths(line))}[/]"
                for line in step.args_hint.splitlines()
            ),
            title="[bold]Command[/]",
            border_style="green",
            padding=(0, 1),
        ))

        # Outputs
        console.print()
        console.print(Panel(
            "\n".join(
                f"  [green]{escape(_expand_paths(line))}[/]"
                for line in step.output_hint.splitlines()
            ),
            title="[bold]Outputs[/]",
            border_style="dim",
            padding=(0, 1),
        ))

        # Advice
        if step.advice:
            console.print()
            console.print(Panel(
                f"[italic]{step.advice}[/]",
                title="[bold yellow]Advice[/]",
                border_style="yellow",
                padding=(0, 1),
            ))

    def show_experiments() -> None:
        header()
        console.print(Rule("[bold cyan]Training Experiments[/]", style="cyan"))
        console.print(
            "\n[dim]After preprocessing, run training with:[/]\n"
            "  [bold green]python src/train.py experiment=<name>[/]\n"
        )

        table = Table(box=box.ROUNDED, border_style="blue", header_style="bold cyan", expand=False)
        table.add_column("Experiment", style="bold white", min_width=40)
        table.add_column("Description", style="dim")

        for exp, desc in EXPERIMENTS:
            table.add_row(exp, desc)

        console.print(table)
        console.print()
        console.print(
            "[dim]Tip: override the split file on the command line, e.g.:\n"
            "  python src/train.py experiment=yield_africa_fusion_spatial \\\n"
            "    data.saved_split_file_name=split_spatial_25km.pth[/]"
        )

    def run_step_prompt(step: PipelineStep) -> None:
        console.print()
        show_step_detail(step)
        console.print()

        if not Confirm.ask(f"[bold]Run [yellow]{step.script}[/] now?[/]", default=False):
            return

        # Build command — required args first, then interactively prompt for optionals
        base_cmd, optional_args = _parse_optional_args(step)
        extras: list[str] = []
        if optional_args:
            console.print()
            console.print("[bold yellow]Optional arguments[/]")
            for opt in optional_args:
                if opt.is_path:
                    exists_note = (
                        Text(" (file exists)", style="green")
                        if Path(opt.value).exists()
                        else Text(" (not found)", style="dim")
                    )
                    label = Text.assemble(
                        "Include ", (opt.flag, "bold cyan"), "?\n  ",
                        (opt.value, "dim"), exists_note,
                    )
                    if Confirm.ask(label, default=False):
                        extras.append(f"{opt.flag} {opt.value}")
                elif opt.value:
                    label = Text.assemble(
                        (opt.flag, "bold cyan"),
                        ("  (Enter to use default, type new value, or leave blank to skip)", "dim"),
                    )
                    ans = Prompt.ask(label, default=opt.value)
                    if ans and ans != "skip":
                        extras.append(f"{opt.flag} {ans}")
                else:
                    label = Text.assemble("Include flag ", (opt.flag, "bold cyan"), "?")
                    if Confirm.ask(label, default=False):
                        extras.append(opt.flag)

        final_cmd = base_cmd + (" " + " ".join(extras) if extras else "")
        console.print()
        user_cmd = Prompt.ask("[bold green]Command to run[/]", default=final_cmd)
        console.print()
        console.print(f"[dim]Running:[/] [bold]{escape(user_cmd)}[/]")
        console.print(Rule(style="dim"))

        proc = subprocess.Popen(user_cmd, shell=True, start_new_session=True)
        try:
            ret = proc.wait()
            console.print(Rule(style="dim"))
            if ret == 0:
                console.print("[bold green]Step completed successfully.[/]")
            else:
                console.print(f"[bold red]Step exited with code {ret}.[/]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — terminating subprocess…[/]")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except (ProcessLookupError, ChildProcessError):
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, ChildProcessError):
                    pass
                proc.wait()

    # -----------------------------------------------------------------------
    # Main menu loop
    # -----------------------------------------------------------------------
    while True:
        console.clear()
        header()
        console.print()

        console.print(Panel(
            "[bold white]Select an option:[/]\n\n"
            "  [bold cyan]1[/]  Pipeline overview & status\n"
            "  [bold cyan]2[/]  Environment variables\n"
            "  [bold cyan]3[/]  Step 0  — AI-readiness analysis (raw data)  [dim](optional)[/]\n"
            "  [bold cyan]4[/]  Step 1  — Build model-ready CSV  [bold yellow](required)[/]\n"
            "  [bold cyan]5[/]  Step 2a — Augment with NDVI\n"
            "  [bold cyan]6[/]  Step 2b — Augment with AgERA5 climate\n"
            "  [bold cyan]7[/]  Step 2c — Download TESSERA embeddings\n"
            "  [bold cyan]8[/]  Step 2d — Merge augmented CSVs             [dim](optional)[/]\n"
            "  [bold cyan]9[/]  Step 2e — Compare all augmentations        [dim](optional)[/]\n"
            "  [bold cyan]a[/]  Step 3a — Generate spatial splits\n"
            "  [bold cyan]b[/]  Step 3b — Generate LOCO splits\n"
            "  [bold cyan]c[/]  Training experiments reference\n"
            "  [bold cyan]q[/]  Quit",
            title="[bold cyan]Menu[/]",
            border_style="cyan",
            padding=(0, 2),
        ))
        console.print()

        choice = Prompt.ask("[bold]Choice[/]", choices=["1","2","3","4","5","6","7","8","9","a","b","c","q"], show_choices=False)

        if choice == "q":
            console.print("\n[dim]Goodbye.[/]\n")
            break

        console.clear()

        if choice == "1":
            show_overview()
        elif choice == "2":
            show_env()
        elif choice == "3":
            run_step_prompt(STEP_INDEX["ai_readiness"])
        elif choice == "4":
            run_step_prompt(STEP_INDEX["make_ready"])
        elif choice == "5":
            run_step_prompt(STEP_INDEX["augment_ndvi"])
        elif choice == "6":
            run_step_prompt(STEP_INDEX["augment_agera5"])
        elif choice == "7":
            run_step_prompt(STEP_INDEX["tessera"])
        elif choice == "8":
            run_step_prompt(STEP_INDEX["merge_augmentations"])
        elif choice == "9":
            run_step_prompt(STEP_INDEX["check_augmentations"])
        elif choice == "a":
            run_step_prompt(STEP_INDEX["spatial_splits"])
        elif choice == "b":
            run_step_prompt(STEP_INDEX["loco_splits"])
        elif choice == "c":
            show_experiments()

        console.print()
        input("Press Enter to return to the menu…")


# ===========================================================================
# PLAIN ANSI UI  (fallback when rich is not installed)
# ===========================================================================

def _box(title: str, lines: List[str], width: int = 72, border: str = FG_CYAN) -> str:
    inner = width - 4
    top    = border + "╔" + "═" * (width - 2) + "╗" + RESET
    bottom = border + "╚" + "═" * (width - 2) + "╝" + RESET
    bar    = border + "║" + RESET
    title_line = bar + " " + BOLD + FG_YELLOW + title.ljust(inner) + RESET + " " + bar

    body_lines = [top, title_line, border + "╟" + "─" * (width - 2) + "╢" + RESET]
    for line in lines:
        # truncate to inner width
        visible = line  # may contain ANSI; approximate
        padded = line + " " * max(0, inner - len(_strip_ansi(line)))
        body_lines.append(bar + " " + padded + " " + bar)
    body_lines.append(bottom)
    return "\n".join(body_lines)

def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)

def _thin_box(lines: List[str], width: int = 72, border: str = FG_BLUE) -> str:
    inner = width - 4
    top    = border + "┌" + "─" * (width - 2) + "┐" + RESET
    bottom = border + "└" + "─" * (width - 2) + "┘" + RESET
    bar    = border + "│" + RESET

    body = [top]
    for line in lines:
        padded = line + " " * max(0, inner - len(_strip_ansi(line)))
        body.append(bar + " " + padded + " " + bar)
    body.append(bottom)
    return "\n".join(body)

def _rule(width: int = 72, colour: str = FG_CYAN) -> str:
    return colour + "─" * width + RESET

def _header_ansi() -> None:
    w = min(_term_width(), 80)
    print()
    title = (
        BOLD + FG_GREEN + "AETHER-xAI" + RESET +
        FG_CYAN + "  |  " + RESET +
        BOLD + FG_WHITE + "Crop Yield Africa" + RESET +
        FG_CYAN + "  |  " + RESET +
        BOLD + FG_YELLOW + "Preprocessing Pipeline" + RESET
    )
    sub = DIM + "Interactive guide to prepare data for model training" + RESET
    print(_box("", [title, sub], width=w, border=FG_CYAN))

def _menu_ansi() -> str:
    w = min(_term_width(), 80)
    lines = [
        BOLD + FG_CYAN + "  1" + RESET + "  Pipeline overview & status",
        BOLD + FG_CYAN + "  2" + RESET + "  Environment variables",
        BOLD + FG_CYAN + "  3" + RESET + "  Step 0  — AI-readiness analysis (raw data)  " + DIM + "(optional)" + RESET,
        BOLD + FG_CYAN + "  4" + RESET + "  Step 1  — Build model-ready CSV  " + BOLD + FG_YELLOW + "(required)" + RESET,
        BOLD + FG_CYAN + "  5" + RESET + "  Step 2a — Augment with NDVI",
        BOLD + FG_CYAN + "  6" + RESET + "  Step 2b — Augment with AgERA5 climate",
        BOLD + FG_CYAN + "  7" + RESET + "  Step 2c — Download TESSERA embeddings",
        BOLD + FG_CYAN + "  8" + RESET + "  Step 2d — Merge augmented CSVs             " + DIM + "(optional)" + RESET,
        BOLD + FG_CYAN + "  9" + RESET + "  Step 2e — Compare all augmentations        " + DIM + "(optional)" + RESET,
        BOLD + FG_CYAN + "  a" + RESET + "  Step 3a — Generate spatial splits",
        BOLD + FG_CYAN + "  b" + RESET + "  Step 3b — Generate LOCO splits",
        BOLD + FG_CYAN + "  c" + RESET + "  Training experiments reference",
        BOLD + FG_CYAN + "  q" + RESET + "  Quit",
    ]
    return _box("Menu", lines, width=w, border=FG_CYAN)

def _overview_ansi() -> None:
    w = min(_term_width(), 80)
    print(_rule(w))
    print(BOLD + FG_CYAN + " Pipeline Overview" + RESET)
    print(_rule(w))
    print()

    col_w = [4, 30, 11, 12]
    header = (
        FG_CYAN + BOLD +
        "#".ljust(col_w[0]) +
        "Step".ljust(col_w[1]) +
        "Required".ljust(col_w[2]) +
        "Status".ljust(col_w[3]) +
        RESET
    )
    print("  " + header)
    print("  " + FG_BLUE + "─" * (sum(col_w)) + RESET)

    for step in STEPS:
        req = (BOLD + FG_YELLOW + "YES" + RESET) if step.required else (DIM + "no" + RESET)
        exists = _check_output_exists(step)
        if exists:
            status = FG_GREEN + "✓ done" + RESET
        elif step.required:
            status = BOLD + FG_YELLOW + "● required" + RESET
        else:
            status = DIM + "○ optional" + RESET

        row = (
            DIM + step.id[:3].ljust(col_w[0]) + RESET +
            BOLD + step.title.ljust(col_w[1]) + RESET +
            req.ljust(col_w[2]) +
            status
        )
        print("  " + row)

    print()
    print(DIM + "  Step 2a/2b/2c are independent and can run in any order after step 1." + RESET)
    print(DIM + "  Step 2d merges augmented CSVs; run after 2a/2b as needed." + RESET)
    print(DIM + "  Steps 3a/3b need step 1 first; they can run in parallel with step 2." + RESET)

def _env_ansi() -> None:
    w = min(_term_width(), 80)
    print(_rule(w))
    print(BOLD + FG_CYAN + " Environment Variables" + RESET)
    print(_rule(w))
    print()

    env = _check_env()
    for k, v in env.items():
        if v:
            val_str = FG_GREEN + v + RESET
        else:
            style = FG_RED if k in ("PROJECT_ROOT", "DATA_DIR") else DIM
            val_str = style + "not set" + RESET
        print(f"  {BOLD}{k:<30}{RESET} {val_str}")

    print()
    required_missing = [k for k in ("PROJECT_ROOT", "DATA_DIR") if not env.get(k)]
    if required_missing:
        print(FG_RED + BOLD + "  WARNING: " + RESET + FG_RED +
              "Required variables not set: " + ", ".join(required_missing) + RESET)
        print(DIM + "  Edit your .env file — this TUI loads it automatically on startup." + RESET)
    else:
        print(FG_GREEN + "  All required env vars are set." + RESET)

def _step_detail_ansi(step: PipelineStep) -> None:
    w = min(_term_width(), 80)
    print(_rule(w))
    print(BOLD + FG_CYAN + f" {step.title}" + RESET)
    print(_rule(w))
    print()

    # What it does
    desc_lines = step.description.splitlines()
    print(_thin_box(desc_lines, width=w, border=FG_BLUE))
    print()

    if step.depends_on:
        print(FG_YELLOW + "  Depends on: " + RESET + ", ".join(step.depends_on))
        print()

    if step.extra_deps:
        print(BOLD + FG_YELLOW + "  Optional extra packages:" + RESET)
        for d in step.extra_deps:
            print(f"  {DIM}•{RESET} {d}")
        print()

    # Command
    cmd_lines = [BOLD + FG_GREEN + "uv run python " + step.script + RESET]
    for line in _expand_paths(step.args_hint).splitlines():
        cmd_lines.append("  " + FG_CYAN + line + RESET)
    print(_thin_box(cmd_lines, width=w, border=FG_GREEN))
    print()

    # Outputs
    out_lines = [FG_GREEN + line + RESET for line in _expand_paths(step.output_hint).splitlines()]
    print(_thin_box(["Outputs:"] + out_lines, width=w, border=DIM + FG_WHITE))
    print()

    # Advice
    if step.advice:
        adv_lines = [DIM + FG_YELLOW + "Advice" + RESET] + [
            DIM + line + RESET for line in step.advice.splitlines()
        ]
        print(_thin_box(adv_lines, width=w, border=FG_YELLOW))
        print()

def _run_step_ansi(step: PipelineStep) -> None:
    _step_detail_ansi(step)

    ans = input(f"\nRun {step.script} now? [y/N] ").strip().lower()
    if ans != "y":
        return

    base_cmd, optional_args = _parse_optional_args(step)
    extras = _prompt_optional_args_plain(optional_args)
    final_cmd = base_cmd + extras

    print(DIM + f"\nCommand:\n  {final_cmd}" + RESET)
    print(DIM + "Press Enter to run it, or type a replacement command:" + RESET)
    user_input = input("> ").strip()
    cmd = user_input if user_input else final_cmd

    print()
    print(_rule(min(_term_width(), 80)))
    proc = subprocess.Popen(cmd, shell=True, start_new_session=True)
    try:
        ret = proc.wait()
        print(_rule(min(_term_width(), 80)))
        if ret == 0:
            print(BOLD + FG_GREEN + "Step completed successfully." + RESET)
        else:
            print(BOLD + FG_RED + f"Step exited with code {ret}." + RESET)
    except KeyboardInterrupt:
        print("\n" + FG_YELLOW + "Interrupted — terminating subprocess…" + RESET)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, ChildProcessError):
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, ChildProcessError):
                pass
            proc.wait()

def _experiments_ansi() -> None:
    w = min(_term_width(), 80)
    print(_rule(w))
    print(BOLD + FG_CYAN + " Training Experiments" + RESET)
    print(_rule(w))
    print()
    print(DIM + "  After preprocessing, run training with:" + RESET)
    print("  " + BOLD + FG_GREEN + "python src/train.py experiment=<name>" + RESET)
    print()

    for exp, desc in EXPERIMENTS:
        print(f"  {BOLD}{FG_WHITE}{exp:<45}{RESET} {DIM}{desc}{RESET}")

    print()
    print(DIM + "  Tip: override the split file on the command line, e.g.:" + RESET)
    print(DIM + "    python src/train.py experiment=yield_africa_fusion_spatial \\" + RESET)
    print(DIM + "      data.saved_split_file_name=split_spatial_25km.pth" + RESET)

def run_plain() -> None:
    colour = _supports_color()
    if not colour:
        # Disable ANSI if no terminal
        global RESET, BOLD, DIM, FG_BLACK, FG_RED, FG_GREEN, FG_YELLOW
        global FG_BLUE, FG_MAGENTA, FG_CYAN, FG_WHITE, FG_BRIGHT_WHITE
        global BG_BLUE, BG_CYAN, BG_GREEN
        RESET = BOLD = DIM = ""
        FG_BLACK = FG_RED = FG_GREEN = FG_YELLOW = ""
        FG_BLUE = FG_MAGENTA = FG_CYAN = FG_WHITE = FG_BRIGHT_WHITE = ""
        BG_BLUE = BG_CYAN = BG_GREEN = ""

    dispatch = {
        "1": _overview_ansi,
        "2": _env_ansi,
        "3": lambda: _run_step_ansi(STEP_INDEX["ai_readiness"]),
        "4": lambda: _run_step_ansi(STEP_INDEX["make_ready"]),
        "5": lambda: _run_step_ansi(STEP_INDEX["augment_ndvi"]),
        "6": lambda: _run_step_ansi(STEP_INDEX["augment_agera5"]),
        "7": lambda: _run_step_ansi(STEP_INDEX["tessera"]),
        "8": lambda: _run_step_ansi(STEP_INDEX["merge_augmentations"]),
        "9": lambda: _run_step_ansi(STEP_INDEX["check_augmentations"]),
        "a": lambda: _run_step_ansi(STEP_INDEX["spatial_splits"]),
        "b": lambda: _run_step_ansi(STEP_INDEX["loco_splits"]),
        "c": _experiments_ansi,
    }

    while True:
        os.system("clear" if os.name != "nt" else "cls")
        _header_ansi()
        print()
        print(_menu_ansi())
        print()

        choice = input("  Choice: ").strip().lower()
        if choice == "q":
            print(DIM + "\nGoodbye.\n" + RESET)
            break

        if choice not in dispatch:
            continue

        os.system("clear" if os.name != "nt" else "cls")
        _header_ansi()
        print()
        dispatch[choice]()
        print()
        input("  Press Enter to return to the menu…")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    if HAS_RICH:
        run_rich()
    else:
        run_plain()


if __name__ == "__main__":
    main()
