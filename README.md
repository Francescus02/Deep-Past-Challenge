# 📜 Deep Past Initiative: Akkadian-to-English Translation (v20)

This repository contains a highly optimized inference pipeline for the **Deep Past Initiative** machine translation challenge. The project focuses on translating ancient Akkadian transliterations into English using a two-model ByT5 ensemble and a novel **Cross-Model-Aware Agreement Minimum Bayes Risk (MBR)** decoding strategy.

---

## 🚀 Key Innovation: Source-Aware Agreement MBR

In standard ensemble decoding, candidates are often treated equally regardless of their source. Version 20 introduces **Source-Aware Agreement Scoring**, which distinguishes between two fundamentally different types of evidence:

* **Within-Model Agreement (Weak):** Occurs when a single model repeats a candidate across different beam paths. This often results from tiny token-level perturbations and is a weak signal of ground-truth correctness.
* **Cross-Model Agreement (Strong):** Occurs when two independently fine-tuned models converge on the exact same string via different generation paths. This is treated as a high-confidence signal, analogous to a **Product-of-Experts (PoE)** distribution.

### 🧮 Scoring Formula
The final score for each candidate $h_i$ is calculated as:
$$score(h_i) = Utility_{weighted}(h_i) + (Bonus_{within} \times Extras) + (Bonus_{cross} \times Agreed)$$

* **Cross-Model Bonus (+0.12):** Strong independent confirmation.
* **Within-Model Bonus (+0.015):** Consistency reward for internal repetitions.
* **Metric Fusion:** Utility is calculated as the geometric mean of **sentence-level BLEU** and **chrF++**.

---

## 🛠️ Technical Pipeline

### 1. Akkadian-Specific Preprocessing
The `OptimizedPreprocessor` uses vectorized Pandas operations and Regex to handle:
* **Diacritic Normalization:** Standardizes ASCII digraphs (e.g., `sz` → `š`) and index vowels.
* **Logogram Formatting:** Standardizes Sumerograms and phonetic complements.
* **Unified Gap Handling:** Collapses various scholarly damage notations into a single `<gap>` token.
* **Canonicalization:** Normalizes ancient weights, measures, and fractions (e.g., `0.8333` → `⅚`).

### 2. Candidate Generation
To maximize the chances of cross-model collisions, we generate a diverse pool of **~32 candidates per sentence**:
* **Beam Search:** 4 candidates (Beam size 8, LP 1.3).
* **Multi-Temperature Nucleus Sampling:** 6 candidates across temperatures `[0.6, 0.8, 0.75]`.
* **Epsilon Sampling:** 6 candidates using `epsilon_cutoff=0.02`.

### 3. Model Orchestration (`InferenceEngine`)
* **Sequential Loading:** To stay within the **15.6 GB VRAM** limit of the Tesla T4, Model A is loaded, run, and completely unloaded before Model B is initialized.
* **Bucket-Batching:** Inputs are grouped by length to minimize padding overhead, significantly increasing throughput.
* **Adaptive Beam Sizing:** Beam width is dynamically adjusted based on input sequence density to optimize speed without sacrificing quality.

### 4. Postprocessing & Cleanup
The `VectorizedPostprocessor` refines the English output by:
* **Repetition Removal:** Detecting and collapsing multi-word repeating phrases (a common ByT5 artifact).
* **Historical Terminology:** Converting literal translations into correct historical terms (e.g., `shekels less grains`).
* **Character Filtering:** Safely stripping forbidden symbols while protecting semantic brackets and `<gap>` tags.

---

## 📊 Performance & Evolution
* **v13 (34.3):** Single-model baseline.
* **v17 (35.4):** Two-model ensemble with geometric mean MBR.
* **v20 (35.6):** Implementation of cross-model-aware agreement bonuses.

---

## 🏗️ Requirements & Environment
* **Hardware:** Optimized for NVIDIA Tesla T4 (Kaggle environment).
* **Precision:** Automatic Mixed Precision (**BF16**) for inference speed.
* **Key Libraries:** `transformers`, `torch`, `sacrebleu`, `pandas`, `optimum` (BetterTransformer).

---

## 🔗 Project Links
For the full implementation and interactive notebook, visit the Kaggle project page:
**[Kaggle: Cross-Model-Aware Agreement MBR](https://www.kaggle.com/code/francescocampigotto/cross-model-aware-agreement-mbr)**
