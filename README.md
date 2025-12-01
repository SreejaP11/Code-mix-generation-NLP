# Codemix generation

This repository contains multiple approaches for performing machine translation from **English** to **Hinglish** (a code-mixed form of Hindi and English). The project explores several modeling techniques and evaluates their translation quality using standard NLP metrics.

---

##  Repository Structure

| File | Description |
|-------------|-------------|
| `train.csv` | Training dataset with English-Hinglish sentence pairs |
| `valid.csv` | Validation dataset |
| `ngram.ipynb` | Baseline n-gram model implementation |
| `s2s_lstm_attention.ipynb` | Sequence-to-sequence LSTM model with attention |
| `t5-glove.ipynb` | T5 transformer model with GloVe embeddings |
| `indicbart.ipynb` | Fine-tuned IndicBART model |
| `mt5.py` | Fine-tuned mT5 model |
| `llama.py` | Fine-tuned Llama model |
| `evaluation.ipynb` | Evaluating Llama, IndicBART, mT5 models on MIPE and GLUECoS |
| `translation_results.csv` | Translations from Llama |
| `mt5_translation_results.csv` | Translations from mT5 |
| `CodemixGeneration_PPT.pdf` | Project presentation |
| `CodemixGeneration_Report.pdf` | Project report |
| `README.md` | This documentation file |

---
##  Models Used

###  Baseline:
- **N-gram Language Model** (statistical approach)
- - **Seq2Seq with Attention** (LSTM-based encoder-decoder)

### ✅ Transformer Models:
- **T5 with GloVe Embeddings**
- **IndicBART** (Pretrained for Indic languages)
- **mT5** (Multilingual T5)
- **LLaMA** (with fine-tuning for Hinglish)

---
You're right! For **GLUECoS** evaluation, you'd also include metrics for **Named Entity Recognition (NER)** and **Sentiment Analysis**, alongside those for MIPE. Here's how you can integrate them into your evaluation metrics:

---

##  Evaluation Metrics

Each model is evaluated using the following metrics:

* **BLEU**: Measures n-gram overlap between the generated and reference sentences.

* **SacreBLEU**: A variant of BLEU that ensures consistency and reproducibility of results across different platforms and implementations.

* **chrF**: Evaluates character-level F-score, more suitable for morphologically rich languages.

* **ROUGE**: Focuses on recall-based metrics like longest common subsequence for text generation tasks.

* **Exact Match Accuracy (EM)**: Measures the percentage of predictions that exactly match the reference sentences.

* **NER Consistency**: Evaluates how consistently the Named Entity Recognition (NER) models identify and group entities across English and Hinglish sentences.

* **Sentiment Consistency**: Assesses whether the sentiment predictions for English and Hinglish sentences match, showing alignment in sentiment detection between languages.

These metrics comprehensively assess the model's output quality, including n-gram overlap, character-level similarity, structural alignment, and task-specific evaluations for **NER** and **Sentiment Analysis**.


---

##  Dataset
**Training**
- https://huggingface.co/datasets/findnitai/english-to-hinglish
- large dataset with approx 1,50,000 examples

- train.csv
- small dataset, was used to train the baseline models

**Testing**
- valid.csv
- Columns: `English`, `Hinglish`
- Size: Small to moderate corpus with Hinglish-style informal translations.

  
---

##  Saved models

The saved models are uploaded in the drive link:
https://drive.google.com/drive/folders/1rZbcR4pbXGbiZ-FKwOgIRAwghrWT0cx3?usp=sharing
