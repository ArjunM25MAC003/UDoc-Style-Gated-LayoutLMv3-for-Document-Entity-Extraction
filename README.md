UDoc-Style Gated LayoutLMv3 for Document Entity Extraction
This repository contains a full experimental pipeline for Document Entity Extraction (Token Classification) using LayoutLMv3. It establishes a robust vanilla baseline and introduces a custom, interpretable per-token gated variant inspired by UDoc.

The project is designed to be run end-to-end in a Google Colab environment, complete with training, evaluation, ablations, and an interactive Gradio web interface.

📖 Overview
The core contribution of this project is the Gated LayoutLMv3 Model, which allows for a defensible, controlled comparison of how visual information impacts token classification.

The gated model compares two equally deep streams from the same microsoft/layoutlmv3-base encoder:

Text/Layout-Only Stream: A full encoder forward pass where pixel_values are zeroed out. Text tokens pass through all 12 transformer layers but are strictly deprived of document visual content.

Multimodal Stream: The standard full encoder forward pass utilizing the real page image.

By keeping the encoder depth identical and treating visual information as the only controlled variable, the gating mechanism provides deep interpretability into when and where the model relies on visual cues versus text/layout cues.

✨ Features
End-to-End FUNSD Training: Fully configured training loop (RUN_TINY = False by default for full runs).

Dual Model Comparison: Tests a vanilla LayoutLMv3ForTokenClassification baseline against the custom GatedLayoutLMv3ForTokenClassification.

Rigorous Evaluation: Generates three-seed (42, 1337, 2025) mean ± standard deviation comparison tables to ensure statistical significance.

Granular Metrics: Outputs per-entity-class F1 tables for detailed performance tracking.

Ablation Studies: Includes gate entropy regularization ablation.

Interpretability: Quantitative gate-by-entity analysis coupled with visual token heatmaps to see the gate in action.

Domain Extension: Optional support for the CORD (receipt-domain) dataset.

Interactive Demo: Built-in Gradio upload demo featuring OCR-backed predictions and gate overlays on user-uploaded images.

🛠️ Environment & Installation
This project is optimized for a GPU-accelerated environment (e.g., Google Colab with a T4 GPU).

System Requirements
Tesseract OCR is required for the Gradio upload demo and the optional CORD converter.

Bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
Python Dependencies
Install the required Python packages:

Bash
pip install -U "transformers>=4.42" "datasets>=2.20" "evaluate>=0.4" seqeval accelerate gradio opencv-python matplotlib pandas seaborn tqdm pytesseract
pip install --upgrade --force-reinstall Pillow
🚀 Usage & Configuration
Simply run the notebook cells sequentially. The experiment configuration is centralized at the beginning of the notebook for easy modification.

Default Hyperparameters:

Model Checkpoint: microsoft/layoutlmv3-base

Epochs: 8 (1 if RUN_TINY=True)

Batch Size: 2

Gradient Accumulation: 4

Max Length: 512

Learning Rate: 5e-5

Weight Decay: 0.01

Gate Entropy Lambdas: [0.0, 0.01]

To debug the pipeline quickly without running a full training loop, change the RUN_TINY variable to True.

📊 Datasets
FUNSD (Form Understanding in Noisy Scanned Documents): Used by default. It contains scanned forms with word bounding boxes and token-level entity labels (B-HEADER, B-QUESTION, B-ANSWER, etc.). Its noisy, layout-sensitive nature makes it an excellent benchmark for this model.

CORD: Optional receipt-domain extension included in the experiment pipeline.

🖥️ Interactive Gradio Demo
At the end of the notebook, a Gradio cell will launch an interactive web UI. You can:

Upload an arbitrary document image.

The pipeline will automatically run Tesseract OCR to extract words and bounding boxes.

The custom Gated LayoutLMv3 model will classify the entities.

The output will display predictions along with visual gate overlays (heatmaps) demonstrating which modalities the model prioritized per token.

🤝 Acknowledgments
Built using Hugging Face transformers and datasets.

LayoutLMv3 pre-trained weights provided by Microsoft.

FUNSD dataset hosted by nielsr.
