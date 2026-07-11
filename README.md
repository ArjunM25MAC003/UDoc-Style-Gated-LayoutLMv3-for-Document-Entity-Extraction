# UDoc-Style Gated LayoutLMv3 for Document Entity Extraction

Colab-ready experimental pipeline for document entity extraction using LayoutLMv3. The project compares a standard token-classification baseline against an interpretable, per-token gated variant inspired by UDoc.

The custom gated model blends two equally deep streams from the same encoder:

- Text/layout-only stream: page pixels are zeroed out, so the model sees text tokens and layout through all 12 transformer layers, but no visual content.
- Multimodal stream: the standard LayoutLMv3 forward pass using the real document image.

Because both branches use the same encoder depth, the gating mechanism measures the contribution of visual information without confounding shallow-vs-deep comparisons.

## Repository Contents

- [UDoc_Gated_LayoutLMv3_Colab.ipynb](UDoc_Gated_LayoutLMv3_Colab.ipynb) - main executable Colab notebook.

## Key Highlights

- FUNSD integration with automated loading, inspection, and visual sanity checks.
- Dual models: a vanilla `LayoutLMv3ForTokenClassification` baseline and a custom `GatedLayoutLMv3ForTokenClassification`.
- Rigorous evaluation across three seeds: `42`, `1337`, and `2025`.
- Per-entity-class precision, recall, and F1 analysis.
- Gate entropy regularization ablations with lambdas `0.0` and `0.01`.
- Quantitative gate-value analysis by entity type.
- Token-level gate heatmap visualizations.
- Optional Gradio web UI for OCR-backed document inference.
- Optional CORD receipt-domain extension using the same training and evaluation flow.

## Problem Framing

Document entity extraction depends on both text and layout. This notebook is designed to test how much the model relies on visual cues and to make that dependence visible through token-level gate values.

Interpretation:

- Gate near `1.0`: text/layout-dominant decision.
- Gate near `0.0`: multimodal/vision-dominant decision.

## Datasets

### Primary Benchmark: FUNSD

The main experiment uses FUNSD (Form Understanding in Noisy Scanned Documents), which provides scanned forms with OCR words, normalized bounding boxes, and token-level entity labels. It is a natural fit for LayoutLMv3's multimodal inputs.

### Optional Extension: CORD

The notebook also includes an optional receipt-domain extension for CORD. Since CORD does not match FUNSD's exact `words/bboxes/ner_tags` structure, the notebook builds a weak OCR-backed BIO conversion pipeline before training and evaluation.

## Requirements

Recommended environment:

- Python 3.10+
- Google Colab with GPU enabled, preferably a T4 GPU
- Tesseract OCR for the upload demo and OCR-backed CORD processing

Key Python dependencies installed in the notebook:

- `transformers` >= 4.42
- `datasets` >= 2.20
- `evaluate` >= 0.4
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

## Suggested Run Order

For a complete end-to-end experiment, run the notebook cells in this order:

1. Environment setup.
2. Imports and configuration.
3. FUNSD loading and visualization.
4. Processor and dataset encoding.
5. Metrics and helpers.
6. Baseline training.
7. Gated model training.
8. Comparison tables and plots.
9. Gate analysis and heatmaps.
10. Optional Gradio demo.
11. Optional CORD extension.

## Outputs

The notebook writes experiment artifacts to `/content/udoc_gated_layoutlmv3`.

Expected outputs include:

- `funsd_baseline_seed_runs.csv`
- `funsd_baseline_per_class_seed_runs.csv`
- `funsd_gated_seed_runs.csv`
- `funsd_gated_per_class_seed_runs.csv`
- `funsd_all_seed_runs.csv`
- `funsd_model_comparison_mean_std.csv`
- `funsd_per_class_seed_runs.csv`
- `funsd_per_class_comparison_mean_std.csv`
- `gate_heatmaps/` for overlay images
- `cord_*.csv` files if the CORD extension is run

## Implementation Notes

- `apply_ocr=False` is used for FUNSD training because the dataset already provides OCR words and boxes.
- The gate is token-wise and learned end-to-end.
- The loss supports optional entropy regularization to discourage gate collapse.
- Seed-level reporting is included so results are not based on a single run.
- The Gradio demo uses OCR-backed inference and overlays gate values on the uploaded image.
- Set `RUN_TINY = True` for a fast debugging pass.
- Keep `RUN_CORD_EXPERIMENT = False` until the FUNSD experiment finishes successfully.

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
