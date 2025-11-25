# Earnings Call LLM — Demo Project (Custom Tokenizer, Mini GPT, RAG)

This repository is a **demo implementation** of building a tiny LLM from scratch, it is tiny because of my hardware specs, only run on CPU and low VRAM.  
Even though the compute is limited, this project still demonstrates the **core engineering steps required to build and run a real LLM pipeline**, including:

#### Custom tokenizer (SentencePiece BPE)  
#### Mini GPT-style language model  
#### End-to-end training loop (from scratch)  
#### RAG (Retrieval-Augmented Generation) over earnings call transcripts  
#### CLI tool for training, retrieval, and inference  

The goal:  
> **Learn all components of LLM development by building a complete—though small—working system. It can work like other LLMs like ChatGPT, Gemini, ect., if can be trained through large data**

---

# 📂 Project Structure

```
LLM-Earnings-Call-Analyst/
│
├── src/
│   ├── main.py                 # CLI entrypoint
│   ├── data_utils.py           # Data preprocessing utilities
│   ├── tokenizer_train.py      # Tokenizer training script
│   ├── tokenizer_utils.py      # Tokenizer helper functions
│   ├── train_lm.py             # Language model training
│   ├── model.py                # Model architecture
│   ├── inference_demo.py       # Inference demonstration
│   ├── rag_pipeline.py         # RAG implementation
│   └── config.py               # Configuration management
│
├── data/
│   ├── raw/                    # Earnings transcripts (not uploaded)
│   └── processed/              # Cleaned text split into train/val
│
├── tokenizer/                  # SentencePiece tokenizer files
│
├── checkpoints/                # Model checkpoints (not uploaded)
│
├── config/
│   └── model_config.json       # Model configuration
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```


The following directories intentionally contain **no raw data or heavy checkpoints** due to storage and laptop capacity constraints:
- `data/raw/`
- `checkpoints/`
- `tokenizer/`  
However, the complete **pipeline code is included**, so you can recreate everything.

---

# 🚀 What This Demo Can Do

Even in demo mode, the project lets you:

### **1. Build and train your own tokenizer**
- Byte-Pair Encoding (BPE)
- Adjustable vocab size
- Adjust epoch
### **2. Train a small GPT-like model**
- Embedding sizes from 128–512  
- Context size 128–256  
- 2–6 transformer blocks  
- CPU-friendly tiny model available

### **3. Run RAG (Retrieval Augmented Generation)**
- TF-IDF vectorizer  
- Retrieve top-K most relevant transcript chunks  
- Feed context into your mini-LLM for earnings call analysis  

### **4. Run inference and generate text**
- “Explain what NVIDIA said about data center revenue”  
- “Summarize Apple’s Q3 call”  
- “Predict forward guidance sentiment”  

---

# 🛠 Installation


📦 Installation
===============

```bash
# Clone the repository
git clone https://github.com/<your-username>/LLM-Earnings-Call-Analyst.git
cd LLM-Earnings-Call-Analyst
```

```bash
# Create virtual environment
python -m venv .venv
```

```bash
# Activate environment (Windows)
.venv\Scripts\activate
```

```bash
# Activate environment (macOS/Linux)
source .venv/bin/activate
```

```bash
# Install dependencies
pip install -r requirements.txt
```

---

▶️ Usage Guide (Copy–Paste Ready)
==================================

Everything runs through:

```bash
python src/main.py <command>
```

---

1️⃣ Build the cleaned corpus
----------------------------

```bash
python src/main.py build_corpus
```

---

2️⃣ Train the tokenizer
-----------------------

```bash
python src/main.py train_tokenizer --vocab_size 16000
```

---

3️⃣ Train a tiny language model
-------------------------------

```bash
python src/main.py train_lm --model_size tiny --epochs 1
```

---

4️⃣ Build the RAG index
-----------------------

```bash
python src/main.py build_index
```

---

5️⃣ Ask a question (RAG + LLM)
------------------------------

```bash
python src/main.py ask   --question "What did NVIDIA say about data center demand?"   --ckpt checkpoints/lm_tiny_best.pt
```

---

6️⃣ Generate text from a prompt
------------------------------

```bash
python src/main.py generate   --prompt "NVIDIA expects"   --ckpt checkpoints/lm_tiny_best.pt
```

---

7️⃣ Index a single transcript file
----------------------------------

```bash
python src/main.py index_single --file path/to/transcript.txt
```

---

8️⃣ Full Pipeline (Run Everything)
----------------------------------

```bash
python src/main.py build_corpus
python src/main.py train_tokenizer --vocab_size 16000
python src/main.py train_lm --model_size tiny --epochs 10
python src/main.py build_index
python src/main.py ask --question "What did META say about forward guidance?"
```


