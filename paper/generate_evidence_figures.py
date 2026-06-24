from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageFilter, ImageOps


PAPER_ROOT = Path(__file__).resolve().parent
ANNOTATION_ROOT = PAPER_ROOT.parent

PHASE0_FILE = (
    ANNOTATION_ROOT
    / "results/phase0_zero_shot/canonical_20260425_231347/phase0_zero_shot_results.json"
)
PHASE1_DIR = ANNOTATION_ROOT / "results/phase1_annotations/full_20260418_0001"
PHASE2_DIR = ANNOTATION_ROOT / "results/phase2_filtering/full_20260418_0001"
PHASE3_LINEAR_SUMMARY = (
    ANNOTATION_ROOT
    / "results/phase3_finetune/repeated_seeds/seed43_47_linear_cached_20260504/phase3_repeated_seed_summary.csv"
)
PHASE3_LORA_SUMMARY = (
    ANNOTATION_ROOT
    / "results/phase3_finetune/repeated_seeds/seed43_47_lora_focus20_20260505/phase3_repeated_seed_summary.csv"
)
PHASE3_SINGLE_LORA_DIR = (
    ANNOTATION_ROOT / "results/phase3_finetune/lora_sweep/20260421_235410"
)
PHASE3_QWEN35_DIR = (
    ANNOTATION_ROOT / "results/phase3_finetune/qwen35_27b_none"
)
PHASE4_QWEN35_SELECTIVE = (
    ANNOTATION_ROOT
    / "results/phase4_selective_annotation/qwen35_27b_none/TeacherBehavior_selective_linear_result.json"
)
PHASE4_DEFAULT_SELECTIVE = (
    ANNOTATION_ROOT
    / "results/phase4_selective_annotation/default/TeacherBehavior_selective_linear_result.json"
)
PHASE0_CAPE_FILE = (
    ANNOTATION_ROOT
    / "results/phase0_zero_shot/cape_aux/phase0_cape_zero_shot_setABC_results.json"
)

TEACHERBEHAVIOR_IMAGE_ROOT = Path(
    "/school_Agri/datasets_scb/SCB5_TeacherBehavior/"
    "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2/images"
)

DATASET_ORDER = ["BowTurnHead", "HandriseReadWrite", "TeacherBehavior"]
CONDITION_SPECS = [
    ("ZS", None, "zs"),
    ("None-LP", "none", "linear"),
    ("None-LoRA", "none", "lora"),
    ("Agree-LP", "agreement", "linear"),
    ("Agree-LoRA", "agreement", "lora"),
    ("GT-LP", "gt", "linear"),
    ("GT-LoRA", "gt", "lora"),
]
REPEATED_LORA = {
    ("BowTurnHead", "none"),
    ("BowTurnHead", "gt"),
    ("TeacherBehavior", "none"),
    ("TeacherBehavior", "gt"),
}

# =========================================================
# IEEE Access Modern Publication Palette
# =========================================================

PRIMARY_BLUE = "#376092"
SECONDARY_TEAL = "#4A877E"
ACCENT_GOLD = "#BE8930"

NEUTRAL_FILL_COLOR = "#B7B7B7"

TEXT_COLOR = "#1F2937"
EDGE_COLOR = "#374151"

GRID_COLOR = "#E5E7EB"

BACKGROUND_COLOR = "#FFFFFF"
PANEL_BACKGROUND = "#F9FAFB"

QWEN35_COLOR = "#C44E52"  # Distinct red for Qwen3.5-27B

# =========================================================
# Semantic color mapping (IEEE-style)
# =========================================================

CONDITION_COLORS = {
    "ZS": NEUTRAL_FILL_COLOR,
    "None-LP": SECONDARY_TEAL,
    "None-LoRA": SECONDARY_TEAL,
    "Agree-LP": PRIMARY_BLUE,
    "Agree-LoRA": PRIMARY_BLUE,
    "GT-LP": ACCENT_GOLD,
    "GT-LoRA": ACCENT_GOLD,
}

DATASET_STYLES = {
    "BowTurnHead": {"alpha": 1.0, "hatch": None},
    "HandriseReadWrite": {"alpha": 0.82, "hatch": None},
    "TeacherBehavior": {"alpha": 1.0, "hatch": "///"},
}
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")
PRIVACY_THUMBNAIL_SIZE = (22, 22)
TEACHERBEHAVIOR_SHIFT_ORDER = [
    "teacher", "stand", "guide", "blackboard-writing",
    "blackboard", "screen", "answer", "on-stage interaction",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def configure_paper_figure(font_size: float = 8.0) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": font_size,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.facecolor": BACKGROUND_COLOR,
        "figure.facecolor": BACKGROUND_COLOR,
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 0.6,
    })


def lighten_color(hex_color: str, blend: float = 0.28) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(c * (1.0 - blend) + blend for c in rgb)


def style_axis(ax, grid_axis: str | None = None, tick_fontsize: float = 8.0) -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_color(EDGE_COLOR)
        ax.spines[spine_name].set_linewidth(0.6)
    ax.tick_params(labelsize=tick_fontsize, colors=TEXT_COLOR, width=0.6, length=3.0)


def build_overview_plot_data() -> Dict[str, Dict[str, dict]]:
    zero_shot = {
        item["dataset"]: item["overall_acc"] for item in load_json(PHASE0_FILE)
    }
    linear_summary = pd.read_csv(PHASE3_LINEAR_SUMMARY)
    lora_summary = pd.read_csv(PHASE3_LORA_SUMMARY)

    repeated_linear = {
        (row.dataset, row.pseudo_strategy): {
            "value": float(row.mean_val_acc), "sd": float(row.std_val_acc), "evidence": "repeated",
        }
        for row in linear_summary.itertuples(index=False)
    }
    repeated_lora = {
        (row.dataset, row.pseudo_strategy): {
            "value": float(row.mean_val_acc), "sd": float(row.std_val_acc), "evidence": "replicated",
        }
        for row in lora_summary.itertuples(index=False)
    }

    single_lora: Dict[Tuple[str, str], dict] = {}
    for dataset in DATASET_ORDER:
        for strategy in ("none", "agreement", "gt"):
            result_path = PHASE3_SINGLE_LORA_DIR / f"{dataset}_lora_{strategy}_result.json"
            result = load_json(result_path)
            single_lora[(dataset, strategy)] = {
                "value": float(result["best_val_acc"]), "sd": None, "evidence": "single",
            }

    plot_data: Dict[str, Dict[str, dict]] = {}
    for dataset in DATASET_ORDER:
        dataset_data: Dict[str, dict] = {}
        for condition_label, strategy, mode in CONDITION_SPECS:
            if mode == "zs":
                dataset_data[condition_label] = {
                    "value": float(zero_shot[dataset]), "sd": None, "evidence": "single",
                }
            elif mode == "linear":
                dataset_data[condition_label] = repeated_linear[(dataset, strategy)]
            else:
                if (dataset, strategy) in REPEATED_LORA:
                    dataset_data[condition_label] = repeated_lora[(dataset, strategy)]
                else:
                    dataset_data[condition_label] = single_lora[(dataset, strategy)]
        plot_data[dataset] = dataset_data

    return plot_data


def generate_overview_figure() -> None:
    plot_data = build_overview_plot_data()
    configure_paper_figure(font_size=8.5)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    x = np.arange(len(CONDITION_SPECS))
    width = 0.22
    offsets = {
        "BowTurnHead": -width,
        "HandriseReadWrite": 0.0,
        "TeacherBehavior": width,
    }

    for dataset in DATASET_ORDER:
        positions = x + offsets[dataset]
        dataset_style = DATASET_STYLES[dataset]
        for idx, (condition_label, _, _) in enumerate(CONDITION_SPECS):
            entry = plot_data[dataset][condition_label]
            condition_color = CONDITION_COLORS[condition_label]
            yerr = entry["sd"] if entry["sd"] is not None else None
            error_kw = {"elinewidth": 0.6, "capsize": 2.5, "capthick": 0.6, "ecolor": EDGE_COLOR}
            bar_kwargs = {
                "x": positions[idx], "height": entry["value"], "width": width * 0.95,
                "color": condition_color, "alpha": dataset_style["alpha"],
                "hatch": dataset_style["hatch"], "edgecolor": EDGE_COLOR,
                "linewidth": 0.6, "zorder": 3,
            }
            if yerr is not None:
                bar_kwargs["yerr"] = yerr
                bar_kwargs["error_kw"] = error_kw
            if entry["evidence"] == "single":
                bar_kwargs["linewidth"] = 0.9
                bar_kwargs["edgecolor"] = "#6B7280"
            ax.bar(**bar_kwargs)

    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(["ZS", "None\nLP", "None\nLoRA", "Agree\nLP", "Agree\nLoRA", "GT\nLP", "GT\nLoRA"], rotation=0)
    style_axis(ax, grid_axis="y", tick_fontsize=8.0)

    legend_handles = [
        Patch(facecolor=PRIMARY_BLUE, edgecolor=EDGE_COLOR, linewidth=0.6, label="Agreement-based methods"),
        Patch(facecolor=SECONDARY_TEAL, edgecolor=EDGE_COLOR, linewidth=0.6, label="Weak baselines"),
        Patch(facecolor=ACCENT_GOLD, edgecolor=EDGE_COLOR, linewidth=0.6, label="GT upper-bound"),
        Patch(facecolor=NEUTRAL_FILL_COLOR, edgecolor=EDGE_COLOR, linewidth=0.6, label="Zero-shot reference"),
        Patch(facecolor="#D1D5DB", edgecolor=EDGE_COLOR, linewidth=0.6, label="BowTurnHead"),
        Patch(facecolor="#D1D5DB", edgecolor=EDGE_COLOR, linewidth=0.6, alpha=0.82, label="HandriseReadWrite"),
        Patch(facecolor="#D1D5DB", edgecolor=EDGE_COLOR, hatch="///", linewidth=0.6, label="TeacherBehavior"),
        Line2D([0], [0], color=EDGE_COLOR, linewidth=0.6, marker="|", markersize=10, markeredgewidth=0.6, label="±1 SD"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.22),
              frameon=False, columnspacing=1.2, handletextpad=0.6, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    save_figure(fig, PAPER_ROOT / "fig_all_results_overview_access")


def generate_quality_threshold_figure() -> None:
    """TeacherBehavior: Qwen2-VL-7B vs Qwen3.5-27B pseudo-label quality threshold."""
    zero_shot_cape = load_json(PHASE0_CAPE_FILE)
    cape_tb = None
    for item in zero_shot_cape:
        if item["dataset"] == "TeacherBehavior":
            cape_tb = float(item["overall_acc"])
            break

    qwen2_none_lp = load_json(
        ANNOTATION_ROOT / "results/phase3_finetune/full_pipeline/full_20260418_0001/TeacherBehavior_linear_none_result.json"
    )
    qwen2_none_lora = load_json(
        PHASE3_SINGLE_LORA_DIR / "TeacherBehavior_lora_none_result.json"
    )

    qwen35_none_lp = load_json(PHASE3_QWEN35_DIR / "TeacherBehavior_linear_none_result.json")
    qwen35_none_lora = load_json(PHASE3_QWEN35_DIR / "TeacherBehavior_lora_none_result.json")

    qwen2_selective_lp = load_json(PHASE4_DEFAULT_SELECTIVE)
    qwen35_selective_lp = load_json(PHASE4_QWEN35_SELECTIVE)

    conditions = ["None\nLP", "None\nLoRA", "Selective\nLP"]
    qwen2_values = [
        float(qwen2_none_lp["best_val_acc"]),
        float(qwen2_none_lora["best_val_acc"]),
        float(qwen2_selective_lp["best_overall_acc"]),
    ]
    qwen35_values = [
        float(qwen35_none_lp["best_val_acc"]),
        float(qwen35_none_lora["best_val_acc"]),
        float(qwen35_selective_lp["best_overall_acc"]),
    ]

    configure_paper_figure(font_size=8.5)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    x = np.arange(len(conditions))
    width = 0.32

    ax.bar(
        x - width / 2, qwen2_values, width * 0.92,
        color=SECONDARY_TEAL, edgecolor=EDGE_COLOR, linewidth=0.6, alpha=0.75,
        label="Qwen2-VL-7B\n(annot. acc. 41.2%)", zorder=3,
    )
    ax.bar(
        x + width / 2, qwen35_values, width * 0.92,
        color=QWEN35_COLOR, edgecolor=EDGE_COLOR, linewidth=0.6, hatch="///",
        label="Qwen3.5-27B\n(annot. acc. 50.3%)", zorder=3,
    )

    ax.axhline(y=cape_tb, color=NEUTRAL_FILL_COLOR, linewidth=1.0, linestyle="--", zorder=2)
    ax.text(len(conditions) - 0.45, cape_tb + 0.8, f"Zero-shot CAPE: {cape_tb:.1f}%",
            fontsize=7.2, color=TEXT_COLOR, va="bottom")

    for i, val in enumerate(qwen2_values):
        ax.text(x[i] - width / 2, val + 0.5, f"{val:.1f}", ha="center", fontsize=7.5, color=TEXT_COLOR)
    for i, val in enumerate(qwen35_values):
        ax.text(x[i] + width / 2, val + 0.5, f"{val:.1f}", ha="center", fontsize=7.5, color=TEXT_COLOR)

    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_title("TeacherBehavior: Annotation Quality Threshold", fontsize=9.2, pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=7.2)
    style_axis(ax, grid_axis="y", tick_fontsize=8.0)
    fig.tight_layout()

    save_figure(fig, PAPER_ROOT / "fig_quality_threshold_access")


def generate_distribution_shift_figure() -> None:
    import matplotlib.gridspec as gridspec
    configure_paper_figure(font_size=8.0)
    fig = plt.figure(figsize=(8.2, 6.5))
    grid = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.35], hspace=0.5, wspace=0.35)
    axes = {
        "BowTurnHead": fig.add_subplot(grid[0, 0]),
        "HandriseReadWrite": fig.add_subplot(grid[0, 1]),
        "TeacherBehavior": fig.add_subplot(grid[1, :]),
    }

    for dataset in DATASET_ORDER:
        df = pd.read_csv(PHASE2_DIR / f"{dataset}_train_distribution_shift.csv")
        reference_color = lighten_color(NEUTRAL_FILL_COLOR, blend=0.12)
        agreement_color = lighten_color(PRIMARY_BLUE, blend=0.15)
        if dataset == "TeacherBehavior":
            order_map = {name: i for i, name in enumerate(TEACHERBEHAVIOR_SHIFT_ORDER)}
            df["plot_order"] = df["class_name"].map(order_map).fillna(len(order_map))
            df = df.sort_values("plot_order", kind="stable").drop(columns=["plot_order"])
        else:
            df = df.sort_values("agreement_minus_all_pct", ascending=False)
        df = df.reset_index(drop=True)
        ax = axes[dataset]
        y = np.arange(len(df))
        ax.barh(y - 0.18, df["all_pct"], height=0.34, color=reference_color,
                edgecolor=EDGE_COLOR, linewidth=0.5, alpha=0.82)
        ax.barh(y + 0.18, df["agreement_pct"], height=0.34, color=agreement_color,
                edgecolor=EDGE_COLOR, linewidth=0.5, hatch="///", alpha=0.94)
        max_pct = max(df["all_pct"].max(), df["agreement_pct"].max())
        for idx, row in df.iterrows():
            delta = float(row["agreement_minus_all_pct"])
            ax.text(max(row["all_pct"], row["agreement_pct"]) + 1.0, y[idx] + 0.18,
                    f"{delta:+.1f} pp", va="center", fontsize=7, color=TEXT_COLOR)
        ax.set_title(dataset, fontsize=8.8, pad=6)
        ax.set_yticks(y)
        ax.set_yticklabels(df["class_name"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, max_pct + 14.0)
        style_axis(ax, grid_axis="x", tick_fontsize=7.5)

    axes["TeacherBehavior"].set_xlabel("Class share of the training split (%)")
    axes["BowTurnHead"].legend(
        handles=[
            Patch(facecolor=lighten_color(NEUTRAL_FILL_COLOR, blend=0.12),
                  edgecolor=EDGE_COLOR, linewidth=0.5, label="Full train set"),
            Patch(facecolor=lighten_color(PRIMARY_BLUE, blend=0.15),
                  edgecolor=EDGE_COLOR, hatch="///", linewidth=0.5, label="Agreement subset"),
        ],
        loc="upper center", bbox_to_anchor=(1.1, 1.35), frameon=False, ncol=2, fontsize=7.8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, PAPER_ROOT / "fig_distribution_shift_access")


def normalize_label(label: str) -> str:
    return label.replace("-", "-\n") if len(label) > 14 else label


def agreement_accuracy_by_class(records: Iterable[dict]) -> Dict[str, float]:
    summary: Dict[str, dict] = {}
    for record in records:
        if not record.get("agreement"):
            continue
        gt_name = record["gt_name"]
        if gt_name not in summary:
            summary[gt_name] = {"correct": 0, "total": 0}
        summary[gt_name]["total"] += 1
        summary[gt_name]["correct"] += int(record["pred_qwen_name"] == gt_name)
    return {
        key: (100.0 * value["correct"] / value["total"] if value["total"] else 0.0)
        for key, value in summary.items()
    }


def find_image_path(filename: str) -> Path:
    for split in ("val", "train"):
        candidate = TEACHERBEHAVIOR_IMAGE_ROOT / split / filename
        if candidate.exists():
            return candidate
    for candidate in TEACHERBEHAVIOR_IMAGE_ROOT.rglob(filename):
        return candidate
    raise FileNotFoundError(filename)


def pixelate_for_privacy(crop: Image.Image) -> Image.Image:
    thumbnail = crop.resize(PRIVACY_THUMBNAIL_SIZE, resample=BILINEAR)
    pixelated = thumbnail.resize(crop.size, resample=NEAREST)
    return pixelated.filter(ImageFilter.GaussianBlur(radius=0.6))


def extract_crop(record: dict, output_size: Tuple[int, int] = (260, 260)) -> Image.Image:
    image = Image.open(find_image_path(record["image"])).convert("RGB")
    width, height = image.size
    bbox_width = record["w"] * width
    bbox_height = record["h"] * height
    center_x = record["cx"] * width
    center_y = record["cy"] * height
    pad = 0.18 * max(bbox_width, bbox_height)
    left = max(0, int(round(center_x - bbox_width / 2.0 - pad)))
    top = max(0, int(round(center_y - bbox_height / 2.0 - pad)))
    right = min(width, int(round(center_x + bbox_width / 2.0 + pad)))
    bottom = min(height, int(round(center_y + bbox_height / 2.0 + pad)))
    crop = image.crop((left, top, right, bottom))
    fitted = ImageOps.fit(crop, output_size, method=LANCZOS)
    return pixelate_for_privacy(fitted)


def choose_example_records(records: List[dict], category: str, require_correct: bool) -> List[dict]:
    candidates = [r for r in records if r["gt_name"] == category and r.get("agreement")]
    filtered = [r for r in candidates if (r["pred_qwen_name"] == category) == require_correct]
    ranked = filtered if len(filtered) >= 2 else candidates
    ranked = sorted(ranked, key=lambda r: r["w"] * r["h"], reverse=True)
    selected: List[dict] = []
    seen_images = set()
    for record in ranked:
        if record["image"] in seen_images:
            continue
        selected.append(record)
        seen_images.add(record["image"])
        if len(selected) == 2:
            break
    if len(selected) < 2:
        for record in ranked:
            if record in selected:
                continue
            selected.append(record)
            if len(selected) == 2:
                break
    return selected


def generate_visual_anchoring_examples() -> None:
    records = load_jsonl(PHASE1_DIR / "TeacherBehavior_val_annotations.jsonl")
    acc_map = agreement_accuracy_by_class(records)
    strong_anchor_tint = lighten_color(PRIMARY_BLUE, blend=0.84)
    difficult_anchor_tint = lighten_color(ACCENT_GOLD, blend=0.84)
    category_specs = [
        ("blackboard-writing", True, "Strong visual anchors"),
        ("screen", True, "Strong visual anchors"),
        ("answer", False, "Intent-dependent categories"),
        ("guide", False, "Intent-dependent categories"),
    ]

    crops: Dict[str, List[Image.Image]] = {}
    for category, require_correct, _ in category_specs:
        chosen = choose_example_records(records, category, require_correct=require_correct)
        crops[category] = [extract_crop(record) for record in chosen]

    configure_paper_figure(font_size=8.0)
    fig, axes = plt.subplots(2, 4, figsize=(8.35, 5.75))
    for col, (category, _, _) in enumerate(category_specs):
        for row in range(2):
            axes[row, col].imshow(crops[category][row])
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor(EDGE_COLOR)
        axes[0, col].set_title(
            f"{normalize_label(category)}\nAgreement acc. {acc_map.get(category, 0.0):.1f}%",
            fontsize=7.8, pad=7,
        )

    fig.lines.append(Line2D([0.5, 0.5], [0.12, 0.875], transform=fig.transFigure,
                             color=GRID_COLOR, linewidth=0.6))
    fig.text(0.26, 0.982, "Strong visual anchors", ha="center", va="top", fontsize=8.8,
             bbox={"boxstyle": "square,pad=0.25", "facecolor": strong_anchor_tint,
                   "edgecolor": PRIMARY_BLUE, "linewidth": 0.6})
    fig.text(0.76, 0.982, "Intent-dependent categories", ha="center", va="top", fontsize=8.8,
             bbox={"boxstyle": "square,pad=0.25", "facecolor": difficult_anchor_tint,
                   "edgecolor": ACCENT_GOLD, "linewidth": 0.6})
    fig.text(0.26, 0.948, "blackboard-writing, screen", ha="center", va="top", fontsize=7.3)
    fig.text(0.76, 0.948, "answer, guide", ha="center", va="top", fontsize=7.3)
    fig.text(0.022, 0.69, "Example 1", rotation=90, fontsize=7.2, va="center")
    fig.text(0.022, 0.26, "Example 2", rotation=90, fontsize=7.2, va="center")
    fig.tight_layout(rect=(0.045, 0.02, 0.99, 0.9))
    save_figure(fig, PAPER_ROOT / "fig_visual_anchoring_examples_access")


def main() -> None:
    generate_overview_figure()
    generate_quality_threshold_figure()
    generate_distribution_shift_figure()
    generate_visual_anchoring_examples()


if __name__ == "__main__":
    main()