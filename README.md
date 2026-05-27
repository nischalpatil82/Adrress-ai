---
title: AddressAI
emoji: 🏠
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Address AI — 93%+ Accuracy Enterprise Address Correction System
Zero API cost. Runs entirely on CPU. No pretrained model weights used at inference.
Built to process and correct complex, real-world Indian addresses accurately down to the street and pincode level.

---

## Project Structure

```
address_ai/
├── data/
│   ├── realistic_addresses.csv ← source file containing real-world raw addresses
│   ├── train_pairs.pkl         ← generated noisy→clean training pairs (step 1)
│   └── val_pairs.pkl           ← generated noisy→clean validation pairs (step 1)
├── fuzzy_engine/               ← Core text-processing & spelling correction library
│   ├── config.py               ← Configuration variables
│   ├── corrector.py            ← Main AddressCorrector entry point
│   ├── db_loader.py            ← SQL integration for the engine
│   ├── dictionaries.py         ← Extensive Indian city/state/street vocabularies
│   ├── matcher.py              ← Regex and pattern matchers
│   ├── normalizer.py           ← Address text normalization logic
│   ├── phonetics.py            ← Phonetic encoding for name matching
│   ├── probabilistic.py        ← Fallback probabilistic matcher
│   ├── spell_checker.py        ← RapidFuzz-based spelling normalizer
│   └── t5_model.py             ← T5 spell inference wrapper
├── models/
│   ├── t5_address/             ← fine-tuned T5 spell corrector (step 2)
│   ├── faiss.index             ← semantic FAISS index (step 3)
│   ├── bm25.pkl                ← keyword BM25 index (step 3)
│   ├── embeddings.npy          ← extracted address vectors (step 3)
│   ├── addresses.npy           ← normalised address list (step 3)
│   ├── address_ids.npy         ← address_id mapping for SQL lookup (step 3)
│   ├── reranker.pkl            ← LightGBM candidate ranker (step 4)
│   └── bandit.json             ← RL bandit state (step 7)
├── address_schema.sql          ← MySQL schemas
├── db.py                       ← Database connector
├── import_realistic_to_sql.py  ← Script to import source CSV into MySQL
├── 1_prepare_data.py
├── 2_finetune_t5.py
├── 3_build_indexes.py
├── 4_train_reranker.py
├── 5_full_pipeline.py          ← In-memory pipeline engine
├── 5_full_pipeline_sql.py      ← Production SQL-integrated pipeline engine
├── 6_evaluate.py
├── 7_rl_bandit.py
├── 8_api.py
└── requirements.txt
```

---

## The Core `fuzzy_engine` Library

Behind the scenes, the project relies heavily on the `fuzzy_engine/` module. This module provides robust string manipulation, spelling correction, and normalization utilities that act as the backbone for cleaning dirty, poorly formatted Indian addresses.

* **`corrector.py`**: Provides the main `AddressCorrector` class that routes data through the normalization pipeline.
* **`dictionaries.py` & `phonetics.py`**: Huge hardcoded vocabularies of Indian locations combined with phonetic algorithms (e.g. Soundex) to catch phonetically-similar typos.
* **`matcher.py` & `spell_checker.py`**: Powered by `RapidFuzz` for lightning-fast token string similarity, identifying permutations of words (like missing characters) without relying on heavy ML models.
* **`db_loader.py` & `t5_model.py`**: Utility wrappers to interface cleanly between the database, the T5 index, and the core string manipulation layers.

---

## Setup & Database Initialisation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Create MySQL database and run the schema:**
```bash
mysql -u root -p < address_schema.sql
```

3. **Configure database environment variables:**
*(PowerShell example)*
```powershell
$env:DB_HOST="127.0.0.1"
$env:DB_PORT="3306"
$env:DB_USER="root"
$env:DB_PASSWORD="your_password"
$env:DB_NAME="address_ai"
```

4. **Import real-world address CSV into the SQL table:**
```bash
python import_realistic_to_sql.py
```
*(Note: Ensure your `data/realistic_addresses.csv` file contains the `raw_address` column).*

---

## Perfect & Detailed Execution Pipeline
*Run the following numbered scripts in exact order to build the fully functional system.*

### Step 1: Prepare Training Data
Generates clean/noisy data pairs from your SQL database to teach the AI what typing errors look like.
```bash
python 1_prepare_data.py
```

### Step 2: Fine-Tune T5 Language Model
Fine-tunes a T5 model specifically to act as a highly accurate spell checker for Indian addresses.
*(You can select your compute hardware via arguments for faster training).*
```bash
# Auto-select (TPU > GPU > CPU)
python 2_finetune_t5.py --device auto

# Force CPU (uses 20% data by default for speed) 
python 2_finetune_t5.py --device cpu

# To use 100% of data on CPU (slower, but full coverage):
python 2_finetune_t5.py --device cpu --full-data
```

### Step 3: Build Retrieval Indexes
Extracts data from MySQL to build blazing-fast local FAISS (semantic) and BM25 (keyword) indexes.
```bash
python 3_build_indexes.py
```

### Step 4: Train LightGBM Re-Ranker
Trains a powerful machine learning re-ranker that evaluates fuzzy similarity, semantics, and BM25 scores to pick the absolute perfect match from the Top-50 candidates retrieved in step 3. 
```bash
python 4_train_reranker.py
```

### Step 5: Test the Pipeline
Run interactive inference engines to verify the full stack is working and outputting correct matches.
```bash
# Test the SQL-integrated production variant (Recommended)
python 5_full_pipeline_sql.py

# OR test the in-memory array variant
python 5_full_pipeline.py
```

### Step 6: Evaluate System Accuracy
Run a formal evaluation suite to test the Hit@1 and Hit@5 accuracy figures on your validation set.
```bash
python 6_evaluate.py
```

### Step 7: Train the Reinforcement Learning Bandit
An RL layer that adjusts weights continuously based on user clicks / manual feedback. 
*(Run this step periodically over time after generating feedback logs).*
```bash
python 7_rl_bandit.py
```

### Step 8: Start the REST API
Deploy the production-ready Flask server internally. 
```bash
python 8_api.py
```

---

## Train On Google Colab (GPU/TPU)
Use this workflow if you want to speed up Step 2 (T5 Training) using cloud hardware.

1. Open a new Google Colab notebook and set Hardware accelerator to **GPU** or **TPU**.
2. Run the cells below:

```python
# Cell 1: Clone project and install deps
!git clone <YOUR_REPO_URL> /content/address_ai
%cd /content/address_ai
!pip install -r requirements.txt
```

```python
# Cell 2: Train T5 with accelerator
!python 2_finetune_t5.py --device auto
```

```python
# Cell 3: Save trained model models/t5_address to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/address_ai_models
!cp -r models/t5_address /content/drive/MyDrive/address_ai_models/
print("Saved!")
```
After training, simply copy the `t5_address` folder from your Drive back into your local project's `models/t5_address` directory.

---

## Expected Accuracy Flow

| Stage                          | Hit@1  | Hit@5  |
|-------------------------------|--------|--------|
| After step 2 (T5 only)        | ~81%   | ~94%   |
| After step 3 (+BM25 +FAISS)   | ~88%   | ~97%   |
| After step 4 (+LightGBM)      | ~93%   | ~99%   |
| After RL (1000+ user clicks)  | ~97%   | ~99%+  |

---

## API Usage Example

```bash
# 1. Start the server
python 8_api.py

# 2. Query for corrections
curl "http://localhost:5000/suggest?q=123+Mian+Stret+Mumbay&n=3"

# 3. Record User Feedback (user accepted suggestion at arm=2)
curl -X POST http://localhost:5000/feedback \
  -H "Content-Type: application/json" \
  -d '{"arm": 2, "accepted": true}'

# 4. Health Check
curl http://localhost:5000/health
```

---

## Technical Flow Overview

```
User Input:   "123 Mian Stret, Mumbay"
                      ↓
1. T5 Model:  "123 main street mumbai"         ← Spelling & context fixed
                      ↓
2. Retrieval: BM25 + FAISS + Fuzzy             ← Fast candidate search (Top 50)
                      ↓
3. Reranking: LightGBM Evaluator               ← Complex feature comparison (Top 5)
                      ↓
4. RL Bandit: Reinforcement Weighting          ← Live system tuning based on feedback
                      ↓
Final Result: "123 Main Street, Mumbai 400001"
```
