# RAPT-CLIP-RAER: Robust Academic Performance Tracking with CLIP for Real-world Academic Emotion Recognition

This repository contains the official implementation of **RAPT-CLIP-RAER**, a state-of-the-art framework for Video Facial Expression Recognition (VFER) and Context-Aware Academic Emotion Recognition. 

Building upon the foundations of CLIP-CAER, this project introduces advanced techniques including **Expression-Aware Adapters (EAA)**, **Temporal Attention Pooling**, **Prompt Learning with Ensembles**, and **Semantic Label Distribution Learning (LDL)** to achieve robust performance in challenging, real-world academic environments.

## 🌟 Key Features

*   **Dual-Stream Architecture:** Simultaneously processes **Face** (fine-grained expression) and **Context/Body** (posture and behavior) streams using a shared CLIP backbone.
*   **Temporal Attention Pooling:** A Transformer-based temporal model that dynamically weights frames to focus on "peak" emotional moments, effectively handling neutral-dominant video sequences.
*   **Expression-Aware Adapter (EAA):** A lightweight, trainable module inserted into the CLIP visual encoder to adapt it for subtle facial expression recognition without destroying pre-trained knowledge.
*   **Robust Prompt Learning:** Utilizes **Prompt Ensembling** and **Learnable Contexts** to generate powerful text classifiers.
*   **Advanced Loss Functions:**
    *   **Mutual Information (MI) Loss:** Aligns learnable prompts with hand-crafted semantic descriptors to prevent overfitting.
    *   **Decorrelation (DC) Loss:** Reduces redundancy in feature dimensions.
    *   **Semantic LDL:** Handles class ambiguity (e.g., "Confusion" vs. "Neutral") by using soft labels based on semantic similarity.
    *   **MoCo (Momentum Contrast):** Optional self-supervised contrastive learning for better feature representation.

---

## 📊 Supported Datasets

The framework is designed to work with multiple datasets, with specific optimizations for academic emotion recognition.

### 1. RAER (Real-world Academic Emotion Recognition)
*   **Focus:** Students in real classroom environments.
*   **Classes (5):** 
    1.  **Neutral** (Calm, attentive)
    2.  **Enjoyment** (Happy, engaged)
    3.  **Confusion** (Puzzled, trying to understand)
    4.  **Fatigue** (Tired, yawning)
    5.  **Distraction** (Looking away, unfocused)
*   **Characteristics:** Highly imbalanced, subtle expressions, significant pose variations.

### 2. DAiSEE (Dataset for Affective States in E-Environments)
*   **Focus:** Engagement levels in e-learning settings.
*   **Classes (4):** Very Low, Low, High, Very High (Engagement).
*   **Characteristics:** Long videos, subtle changes in engagement.

### 3. CK+ (Extended Cohn-Kanade)
*   **Focus:** Lab-controlled posed facial expressions.
*   **Classes (7):** Anger, Contempt, Disgust, Fear, Happy, Sadness, Surprise.

### 4. SFER & CAER
*   **Focus:** Student Facial Expression / Context-Aware Emotion Recognition.
*   **Classes (7):** Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise.

---

## 🏗️ Model Architecture

The `GenerateModel` class (in `models/Generate_Model.py`) orchestrates the following components:

1.  **Visual Encoder (CLIP ViT-B/32):** Extracts spatial features from every frame. The weights are largely frozen to preserve generalization.
2.  **Expression-Aware Adapter (EAA):** A bottleneck module (Linear -> ReLU -> Linear) added to the visual encoder. It learns to extract emotion-specific features (e.g., eyebrow movement, mouth shape).
3.  **Temporal Transformer:** Takes the sequence of frame features (T x D) and applies Self-Attention.
4.  **Attention Pooling:** Instead of a simple average or `[CLS]` token, an attention mechanism calculates a score for each frame. The final video representation is a weighted sum of frame features, prioritizing frames with strong emotional content.
5.  **Text Encoder (Prompt Learner):** Generates class embeddings using learnable context vectors (e.g., `X X X X [CLASS]`) mixed with fixed templates.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.8+
*   PyTorch 1.12+ (Tested with 2.2.2)
*   CUDA 11.x/12.x

### Setup
```bash
# Create environment
conda create -n raer_clip python=3.8
conda activate raer_clip

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install ftfy regex tqdm opencv-python pillow
```

---

## 📂 Data Preparation

The project expects datasets to be organized with an annotation file (txt) and a directory of videos/images.

### Annotation Format
A text file where each line contains:
`path/to/video_or_folder num_frames label_index`

Example (`train.txt`):
```text
RAER/train/Neutral/student01_clip01 45 0
RAER/train/Confusion/student02_clip05 32 2
...
```

### Bounding Boxes
For the Dual-Stream model, you need JSON files containing face and body bounding boxes for every frame (or at least one per clip).
*   `face_bbox.json`: `{"video_name": {"frame_idx.jpg": [x1, y1, x2, y2], ...}}`
*   `body_bbox.json`: Similar structure for body crops.

---

## 🚀 Usage

### Training
The project uses a highly optimized configuration for the RAER dataset, as defined in `train_sh/ablation/raer_full.sh`.

```bash
bash train_sh/ablation/raer_full.sh
```

**Key Training Parameters:**
*   **Backbone:** CLIP ViT-B/16
*   **Batch Size:** 4
*   **Optimizer:** AdamW (LR: 2e-5, Prompt Learner LR: 3e-4)
*   **Loss:** LDAM Loss with Class-Weighted Sampling.
*   **Regularization:** Mutual Information (MI) and Decorrelation (DC) losses enabled after a 5-epoch warmup.
*   **Data:** 16 segments per video, 224x224 image size, including both face and body crops.

### Ablation Studies
We have conducted extensive ablation studies to verify the effectiveness of each component. These scripts are available in `train_sh/ablation/`:
*   `raer_full.sh`: The complete pipeline (EAA + Prompt Tuning + LDAM + MI/DC Loss).
*   `raer_no_adapter.sh`: Removes the Expression-Aware Adapter.
*   `raer_no_prompt_tuning.sh`: Uses fixed hand-crafted prompts instead of learnable contexts.
*   `raer_no_ldam.sh`: Replaces LDAM loss with standard Cross-Entropy.
*   `raer_no_attn-pooling.sh`: Tests alternative temporal aggregation methods.
*   `raer_no_sampler.sh`: Disables the weighted random sampler for handling imbalance.

All ablation experiments have been executed to establish the final "Full" configuration as the best-performing model.

### Evaluation & TTA (Test-Time Augmentation)
To evaluate the best model using Test-Time Augmentation (FiveCrop + Horizontal Flip):

```bash
bash run_tta.sh
```
This script runs `eval_tta.py` which loads the best checkpoint (`model_best.pth`) and performs robust evaluation.

### Ablation Studies
The `train_sh/ablation/` folder contains scripts to test different configurations:
*   `raer_no_adapter.sh`: Train without EAA.
*   `raer_no_prompt_tuning.sh`: Train with fixed prompts.
*   `raer_no_mixup.sh`: Train without Mixup augmentation.

---

## 🌲 Project Structure

```
RAPT-CLIP-RAER/
├── dataloader/          # Custom dataloaders for Video/Image folders
│   ├── video_dataloader.py
│   └── video_transform.py
├── models/              # Model definitions
│   ├── Generate_Model.py      # Main architecture
│   ├── Adapter.py             # EAA module
│   ├── Temporal_Model.py      # Transformer + Attn Pooling
│   ├── Prompt_Learner.py      # Learnable prompts
│   └── Text.py                # Class descriptions
├── train_sh/            # Training shell scripts
│   └── ablation/        # Ablation study configurations
├── utils/               # Helper functions (Loss, Logging)
├── main.py              # Main training loop
├── trainer.py           # Training/Validation engine
├── eval_tta.py          # TTA Evaluation script
└── README.md            # This file
```

## 📜 Citation

If you use this code or dataset, please cite the following paper:

```bibtex
@InProceedings{Zhao_2025_ICCV,
    author    = {Zhao, Luming and Xuan, Jingwen and Lou, Jiamin and Yu, Yonghui and Yang, Wenwu},
    title     = {Context-Aware Academic Emotion Dataset and Benchmark},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2025}
}
```

## 🙏 Acknowledgments
This codebase is built upon [DFER-CLIP](https://github.com/zengqunzhao/DFER-CLIP). We thank the authors for their open-source contribution.