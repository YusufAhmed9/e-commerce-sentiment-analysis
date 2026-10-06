# Baseline Model Evaluation: Sentiment Analysis

## 1. Overview
This report details the baseline performance of our initial sentiment analysis classifier. Establishing a baseline provides a grounded metric against which all future pipeline optimizations (e.g., advanced tokenization, deep learning architectures, lemmatization) will be measured.

* **Task:** Sentiment Classification (Positive, Neutral, Negative)
* **Baseline Model:** [e.g., TF-IDF Vectorizer + Logistic Regression]
* **Text Preprocessing:** [e.g., Basic whitespace tokenization, no stemming, stopword removal applied]

---

## 2. Dataset Statistics
* **Training Set Size:** ~130_000
* **Validation/Test Set Size:** 17_000
* **Class Distribution (Test Set):**
  * Positive: 39.74%
  * Neutral: 21.39%
  * Negative: 38.87%

---

## 3. Performance Metrics

The baseline model was evaluated using standard classification metrics. 

**Overall Accuracy:** **[e.g., 76.5%]**

### Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| **Positive** | [0.70]    | [0.74] | [0.72]   | [6080]   |
| **Neutral**  | [0.61]    | [0.50] | [0.55]   | [3333]   |
| **Negative** | [0.70]    | [0.73] | [0.72]   | [6133]   |
| **Accuracy** |           |        | **[0.68]**| [15546]  |
| **Macro Avg**| [0.67]    | [0.66] | [0.66]   | [15546]  |
| **Wgt Avg**  | [0.68]    | [0.68] | [0.68]   | [15546]  |

*Note: The model currently struggles most with the **[Neutral]** class, likely due to semantic overlap with both positive and negative vocabularies.*

---

## 4. Confusion Matrix Summary
![Confusion Matrix](assets/Screenshot%20From%202026-10-04%2004-05-48.png)
* **Notable Misclassifications:** 
  * 530 Negative samples were incorrectly predicted as Neutral.
  * 110 Positive samples were incorrectly predicted as Negative.

---

## 5. Observations & Next Steps

### Current Limitations
1. **Out-of-Vocabulary (OOV) Terms:** The baseline preprocessing does not handle complex morphology, leading to a fragmented feature space.
2. **Noise in Text:** Emojis and punctuation are currently [handled/unhandled], which may be diluting the signal of strong sentiments.

### Proposed Optimizations (Ablation Plan)
To improve upon this baseline, the following preprocessing and modeling steps will be evaluated sequentially:
1. **Implement Robust Cleaning:** handle emojis/tatweel.
2. **Experiment with Stemming/Lemmatization:** Introduce a light stemmer to reduce feature sparsity and compare the impact on the F1-Score.
3. **Advanced Architectures:** Evaluate transformer-based embeddings (e.g., BERT/AraBERT) once the preprocessing pipeline is maximized.