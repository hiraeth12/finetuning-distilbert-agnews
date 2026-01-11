# Fine-tuning DistilBERT on AG News Dataset

## 👥 Student Identification

### Group 3 Information

| Name | NIM |
|-------|-------------|
| **Sahrul Ridho Firdaus** | 1103223009 |
| **Rayhan Diff** | 1103220039 |


## 📋 Project Overview

This repository contains a deep learning project focused on **text classification** using the AG News dataset. The project implements fine-tuning of the **DistilBERT** model (a distilled version of BERT) for classifying news articles into four distinct categories.

### Purpose

The primary goal of this project is to demonstrate the effectiveness of transfer learning and fine-tuning pre-trained language models for news classification tasks. By leveraging DistilBERT's pre-trained knowledge, we achieve high accuracy while maintaining computational efficiency.

---

## 🎯 Dataset Information

**Dataset:** AG News Dataset (`sh0416/ag_news`)

The AG News dataset consists of news articles categorized into **4 classes**:

| Label ID | Category | Description |
|----------|----------|-------------|
| 0 | World | International news and events |
| 1 | Sports | Sports-related news |
| 2 | Business | Business and finance news |
| 3 | Sci/Tech | Science and technology news |

Each sample in the dataset contains:
- **Title**: News article headline
- **Description**: Brief description/summary of the article
- **Label**: Category classification (0-3)

---

## 🤖 Model Architecture

### Base Model: DistilBERT-base-uncased

**DistilBERT** is a lighter and faster variant of BERT (Bidirectional Encoder Representations from Transformers) that:
- Retains 97% of BERT's language understanding capabilities
- Is 40% smaller in size
- Runs 60% faster
- Uses knowledge distillation during pre-training

### Model Configuration

```python
Model: distilbert-base-uncased
Task: Sequence Classification
Number of Labels: 4
Max Sequence Length: 512 tokens
```

---

## 📊 Training Configuration & Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Learning Rate** | 2e-5 | Adam optimizer learning rate |
| **Batch Size** | 64 | Training and evaluation batch size |
| **Epochs** | 3 | Number of training epochs |
| **Weight Decay** | 0.01 | L2 regularization parameter |
| **Precision** | FP16 | Mixed precision training for faster computation |
| **Optimizer** | AdamW | Default optimizer with weight decay |
| **Eval Strategy** | Epoch | Evaluate after each epoch |
| **Save Strategy** | Epoch | Save checkpoint after each epoch |

---

## 📈 Model Performance & Results

### Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Primary metric for classification performance
- **Loss**: Cross-entropy loss for training optimization

### Expected Performance

Based on the training configuration:
- **Training Time**: Approximately 15-30 minutes (with GPU acceleration)
- **Target Accuracy**: >90% on test set
- **Best Model Selection**: Automatically loads the best performing checkpoint

> **Note**: Actual results may vary based on hardware and training conditions. Run the notebook to see specific metrics.

---

## 🗂️ Repository Structure

```
finetuning-distilbert-agnews/
│
├── finetuning-distilbert-agnews.ipynb   # Main Jupyter notebook with complete implementation
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
│
├── ag_news_model/                        # Training checkpoints (generated after training)
│   ├── checkpoint-xxx/
│   └── ...
│
└── final_agnews/                         # Final trained model (generated after training)
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    └── ...
```

---

## 🚀 How to Navigate & Run the Project

### Prerequisites

Before running the notebook, ensure you have:
- Python 3.8+
- CUDA-compatible GPU (recommended for faster training)
- Jupyter Notebook or Google Colab

### Installation

#### Option 1: Install from requirements.txt (Recommended)

Install all dependencies at once using the provided requirements file:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (with CUDA support)
- Hugging Face Transformers
- Datasets library
- Evaluate metrics
- Accelerate for optimized training
- Scikit-learn
- Jupyter Notebook support
- And other necessary dependencies

#### Option 2: Manual Installation

Alternatively, install packages individually:

```bash
pip install torch transformers datasets evaluate accelerate scikit-learn jupyter -U
```

#### Verify Installation

Check if PyTorch and CUDA are properly installed:

```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
```

### Step-by-Step Guide

1. **Setup & Installation** (Cell 1-2)
   - Install required libraries
   - Import necessary modules

2. **Data Loading** (Cell 3)
   - Load AG News dataset from Hugging Face
   - Fix label indexing (0-3)
   - Verify label distribution

3. **Tokenization** (Cell 4)
   - Initialize DistilBERT tokenizer
   - Preprocess text (title + description)
   - Create tokenized dataset

4. **Model Training** (Cell 5)
   - Configure training arguments
   - Initialize Trainer
   - Fine-tune the model
   - Automatic evaluation per epoch

5. **Save & Test** (Cell 6)
   - Save the final trained model
   - Test inference on sample text
   - Verify predictions

### Running the Notebook

**Option 1: Local Machine**
```bash
jupyter notebook finetuning-distilbert-agnews.ipynb
```

**Option 2: Google Colab**
1. Upload the notebook to Google Drive
2. Open with Google Colab
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run all cells sequentially

### Expected Output

After successful training, you should see:
- Training loss decreasing across epochs
- Evaluation accuracy improving
- Final model saved to `./final_agnews/`
- Test prediction demonstrating correct classification

---

## 🔃Download Pre-trained Models

If you want to skip the training process and use the pre-trained models directly, you can download them from Google Drive:

**🔗 [Download Trained Models from Google Drive](https://drive.google.com/drive/folders/1c36FAYeuR8H2E7bFMKA7txNGLJ5ktDgF?usp=sharing)**

The folder contains:
- `final_agnews/` - Final trained model ready for inference
- `ag_news_model/` - Training checkpoints (optional)

### ❓ How to Use Downloaded Models

1. Download the model folder from the link above
2. Extract to your project directory
3. Load the model for inference:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./final_agnews")
tokenizer = AutoTokenizer.from_pretrained("./final_agnews")
```

---

## 💡 Key Features

✅ **Efficient Fine-tuning**: Uses DistilBERT for faster training without sacrificing performance

✅ **Mixed Precision Training**: FP16 precision for 2x speedup on compatible GPUs

✅ **Automatic Best Model Selection**: Saves the best performing checkpoint

✅ **Combined Text Input**: Concatenates title and description for richer context

✅ **GPU Acceleration**: Optimized for CUDA-enabled devices

✅ **Easy Deployment**: Saved model can be loaded for inference

---

## 🧪 Sample Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./final_agnews")
tokenizer = AutoTokenizer.from_pretrained("./final_agnews")

# Test prediction
text = "Oil prices dropped significantly today as the stock market crashed."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits).item()

id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
print(f"Predicted Category: {id2label[prediction]}")
# Output: Predicted Category: Business
```



## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the project owner.

---


**Last Updated**: January 2026

