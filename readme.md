# VLM-CT-IQA: Fine-Tuning MedGemma for CT Image Quality Assessment

This repository contains the official implementation for fine-tuning and evaluating the `google/medgemma-4b-it` model for the task of Low-Dose CT Image Quality Assessment (IQA). The project utilizes Supervised Fine-Tuning (SFT) with Low-Rank Adaptation (LoRA) to teach the Vision Language Model (VLM) how to assess the perceptual quality of CT scans and provide a numerical score with a detailed reasoning.

## Resources on Hugging Face

The core components of this project—the dataset and the fine-tuned model weights—are publicly available on the Hugging Face Hub:

*   **Fine-tuned Model Weights:** [**MikaJesse/medgemma-4b-it-sftCTIQA**](https://huggingface.co/MikaJesse/medgemma-4b-it-sftCTIQA)
*   **Dataset:** [**MikaJesse/LDCTiqa_png**](https://huggingface.co/datasets/MikaJesse/LDCTiqa_png)

> **Note:** The performance of the open-sourced model weights on Hugging Face has been further improved, thanks to the use of a more effective training dataset.

## Core Scripts

This repository includes three main Python scripts to replicate the workflow:

1.  `trainSFT.py`: The main script to start the Supervised Fine-Tuning process using the provided training and validation data.
2.  `inference_baseline.py`: A script to run batch inference using the original, non-finetuned `google/medgemma-4b-it` model.
3.  `inference_finetuned.py`: A script to run batch inference using the fine-tuned LoRA weights generated from the training script.

## Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/VLM-CT-IQA.git
cd VLM-CT-IQA
```

**2. Install dependencies:**
It is recommended to use a virtual environment (e.g., venv or conda).
```bash
pip install -r requirements.txt
```
This project was developed and tested using a nightly build of PyTorch. While the exact version is pinned in `requirements.txt` for reproducibility, nightly versions can be unstable and may not be available for all systems.
We strongly recommend installing a recent stable version of PyTorch first, by following the official instructions at pytorch.org. This will ensure compatibility with your system's CUDA version. 

**3. Download and organize the data:**
The training/validation/test metadata files (`.jsonl`, `.json`) are already included in the `/data` directory. However, you must manually download the image files.

*   Download the image dataset from [Hugging Face Datasets: MikaJesse/LDCTiqa_png](https://huggingface.co/datasets/MikaJesse/LDCTiqa_png).
*   Create a new directory named `LDCTiqa_png` inside the `data` folder.
*   Place all 1000 PNG images inside this new directory.

Your final directory structure should look like this:
```
VLM-CT-IQA/
│
├── data/
│   ├── LDCTiqa_png/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ... (all 1000 images)
│   ├── testData.json       # Test set info (100 filenames & scores)
│   ├── train.jsonl         # Training set (800 samples)
│   └── val.jsonl           # Validation set (100 samples)
│
├── .gitignore
├── requirements.txt
├── trainSFT.py
├── inference_baseline.py
├── inference_finetuned.py
└── README.md
```

## Usage / Workflow

The project is designed to be run in a sequence: Training -> Inference -> Evaluation.

### Step 1: Fine-Tuning the Model

To start the SFT process with LoRA, run the training script. This script will use `data/train.jsonl` and `data/val.jsonl` for training and validation, respectively. The resulting LoRA weights will be saved to the output directory specified within the script or its configuration.

```bash
python trainSFT.py
```

### Step 2: Running Inference

You can run inference with either the baseline model or your newly fine-tuned model.

#### Baseline Model Inference

This script runs inference on the original `google/medgemma-4b-it` model. By default, it is configured to process all 1000 images from the test set.

```bash
python inference_baseline.py
```
> **Note:** If you wish to run inference on a specific subset of images (e.g., 100 images), you will need to modify the `image_dir` path within the script or its configuration to point to a folder containing only those images.

#### Fine-tuned Model Inference

This script loads the base model and applies the fine-tuned LoRA weights to run inference. Ensure the path to your saved LoRA weights is correctly specified.

```bash
python inference_finetuned.py
```

### Inference Output Format

Both inference scripts will generate a JSON file containing detailed results for each image. The format for each entry is as follows:

```json
{
  "file": "0000.png",
  "score": 3.0,
  "time_sec": 37.57,
  "reasoning": "The image shows excellent detail in the soft tissues with minimal noise...",
  "error": null
}
```
This output can then be used for quantitative evaluation (e.g., calculating SRCC/PLCC against `testData.json`) and qualitative analysis.

## Acknowledgments
The image data and ground truth scores used in this project originate from the **LDCT-IQAC 2023 Grand Challenge**. We are grateful to the organizers for making this valuable dataset available to the research community.