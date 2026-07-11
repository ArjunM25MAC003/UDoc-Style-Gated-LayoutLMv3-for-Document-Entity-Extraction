# UDoc-Style Gated LayoutLMv3 for Document Entity Extraction

Colab-ready notebook for document entity extraction with `LayoutLMv3`, built to compare a standard token-classification baseline against an interpretable gated variant.

The gated model blends two equally deep streams from the same encoder:

- Text/layout-only stream: page pixels are zeroed out, so the model sees tokens and layout but no visual content.
- Multimodal stream: the standard LayoutLMv3 forward pass with the real document image.

Because both branches use the same encoder depth, the gate measures the contribution of visual information instead of confounding it with a shallow-vs-deep comparison.

## Repository Contents

- [UDoc_Gated_LayoutLMv3_Colab.ipynb](UDoc_Gated_LayoutLMv3_Colab.ipynb) - main notebook
- [UDoc_Gated_LayoutLMv3_Colab_README.md](UDoc_Gated_LayoutLMv3_Colab_README.md) - this project guide

## Highlights

- FUNSD loading, inspection, and visual sanity checks
- LayoutLMv3 preprocessing with OCR words, bounding boxes, labels, and images
- Vanilla `LayoutLMv3ForTokenClassification` baseline
- Custom `GatedLayoutLMv3ForTokenClassification`
- Three-seed evaluation with mean and standard deviation reporting
- Per-entity-class precision, recall, and F1 analysis
- Gate entropy regularization ablation
- Gate-value analysis by entity type
- Token-level gate heatmap visualizations
- Optional Gradio demo for OCR-backed document inference
- Optional CORD receipt-domain extension using the same training and evaluation flow

## Problem Framing

Document entity extraction is not only a text problem. Layout and visual structure often matter just as much as the words themselves. This notebook is designed to test how much the model depends on those visual cues, and to make that dependence visible through token-level gate values.

Interpretation:

- Gate close to `1.0`: text/layout-dominant decision
- Gate close to `0.0`: multimodal/vision-dominant decision

## Dataset

### Primary benchmark: FUNSD

The main experiment uses FUNSD, which includes scanned forms with OCR words, normalized bounding boxes, and token-level entity labels. It is a natural fit for LayoutLMv3 because the model can consume both document text and page layout.

### Optional extension: CORD

The notebook also includes an optional receipt-domain extension for CORD. Since CORD does not match FUNSD’s exact `words/bboxes/ner_tags` structure, the notebook builds a weak OCR-backed BIO conversion pipeline before training and evaluation.

## Requirements

Recommended environment:

- Python 3.10+
- Google Colab with GPU enabled
- Tesseract OCR for the upload demo and OCR-backed CORD processing

Key Python packages installed in the notebook:

- `transformers`
- `datasets`
- `evaluate`
- `seqeval`
- `accelerate`
- `gradio`
- `pillow`
- `opencv-python`
- `matplotlib`
- `pandas`
- `seaborn`
- `tqdm`
- `pytesseract`

## Quick Start

1. Open the notebook in Colab.
2. Run the environment cell to install dependencies and Tesseract.
3. Load FUNSD and inspect the sample annotations.
4. Run the processor and encoding cells.
5. Train the vanilla baseline.
6. Train the gated model and entropy ablation runs.
7. Review the summary tables, per-class breakdowns, and gate visualizations.
8. Optionally enable the Gradio demo or the CORD extension.

## Outputs

The notebook writes experiment artifacts to `/content/udoc_gated_layoutlmv3`.

Expected files include:

- `funsd_baseline_seed_runs.csv`
- `funsd_baseline_per_class_seed_runs.csv`
- `funsd_gated_seed_runs.csv`
- `funsd_gated_per_class_seed_runs.csv`
- `funsd_all_seed_runs.csv`
- `funsd_model_comparison_mean_std.csv`
- `funsd_per_class_seed_runs.csv`
- `funsd_per_class_comparison_mean_std.csv`
- gate heatmap images under `gate_heatmaps/`
- optional CORD outputs under similarly named `cord_*.csv` files

## Implementation Notes

- `apply_ocr=False` is used for FUNSD training because the dataset already provides OCR words and boxes.
- The gate is token-wise and learned end-to-end.
- The loss supports optional entropy regularization to discourage gate collapse.
- Seed-level reporting is included so results are not based on a single run.
- The Gradio demo uses OCR-backed inference and overlays gate values on the uploaded image.

## Suggested Run Order

For a full experiment, run the notebook in this order:

1. Environment setup
2. Imports and configuration
3. FUNSD loading and visualization
4. Processor and dataset encoding
5. Metrics and helpers
6. Baseline training
7. Gated model training
8. Comparison tables and plots
9. Gate analysis and heatmaps
10. Optional Gradio demo
11. Optional CORD extension

## Notes

- Full training can take a while, especially on CPU or a small GPU.
- If you only want a fast pipeline check, set `RUN_TINY = True`.
- Keep `RUN_CORD_EXPERIMENT = False` until the FUNSD experiment finishes successfully.
- For the best Gradio experience, train the gated model before launching the demo.

## License

No license file is bundled with the notebook. Add one if you plan to publish or share the project on GitHub.
