# Deep Past Initiative: Akkadian-to-English Translation (Hybrid MBR Ensemble)

This repository contains a highly optimized inference pipeline for the **Deep Past Initiative** machine translation challenge. The project focuses on translating ancient Akkadian transliterations into English using a two-model ByT5 ensemble and a streamlined **Hybrid Minimum Bayes Risk (MBR)** decoding strategy.

---

## Key Innovation: Implicit Consensus via Merged Pools

Earlier versions of this pipeline (up to v40) used a "Cross-Model-Aware" scoring system that applied explicit mathematical bonuses when both models produced the same string. However, testing revealed that explicit bonuses double-counted the consensus signal and overrode pairwise quality scores on close calls when used alongside composite metrics.

Version 41 introduces a **Hybrid MBR** approach that captures cross-model agreement *implicitly*:
* **Merged Pools BEFORE Deduplication:** The raw candidate lists from Model A and Model B are combined into a single pool before any postprocessing or deduplication occurs. 
* **The Implicit Signal:** When both models independently generate the same translation, it mathematically acts as multiple identical references within the combined pool. 
* **Uniform Average:** By calculating a simple uniform pairwise average across the deduplicated pool, candidates that were produced by both models naturally score higher because they are highly similar to a denser cluster of consensus candidates. No explicit bonus is needed; the signal is already in the utility.

### Scoring Formula
The final score for each candidate relies on an additive composite metric and a length tie-breaker, without explicit agreement bonuses:

`PairwiseScore(h_i) = UniformAverage(0.55 * chrF++ + 0.25 * BLEU + 0.20 * Jaccard)`

`FinalScore(h_i) = PairwiseScore(h_i) + (0.10 * GaussianLengthBonus)`

*Note: The Gaussian length bonus is a symmetric bell curve centered on the median pool length, designed to gently nudge scores (by at most 10 points) to break ties without overriding quality signals.*

---

## Technical Pipeline

### 1. Akkadian-Specific Preprocessing
The `OptimizedPreprocessor` uses vectorized Pandas operations and Regex to handle:
* **Diacritic Normalization:** Standardizes ASCII digraphs (e.g., `sz` → `š`) and index vowels.
* **Logogram Formatting:** Standardizes Sumerograms and phonetic complements.
* **Unified Gap Handling:** Collapses various scholarly damage notations into a single `<gap>` token.
* **Canonicalization:** Normalizes ancient weights, measures, and fractions (e.g., `0.8333` → `⅚`).

### 2. Candidate Generation
To maximize the diversity and coverage of the combined pool, we generate **~32 candidates per sentence** by merging a wide temperature spread with targeted epsilon sampling:
* **Beam Search:** 4 candidates (Beam size 8, LP 1.3).
* **Multi-Temperature Nucleus Sampling:** 6 candidates across a wide hybrid temperature spread `[0.55, 0.75, 0.95]`.
* **Epsilon Sampling:** 6 candidates using `epsilon_cutoff=0.02` to inject crucial diversity.

### 3. Model Orchestration (`InferenceEngine`)
* **Sequential Loading:** To stay within the **15.6 GB VRAM** limit of the Tesla T4, Model A is loaded, run, and completely unloaded from the GPU before Model B is initialized.
* **Bucket-Batching:** Inputs are grouped by length to minimize padding overhead, significantly increasing throughput.
* **Adaptive Beam Sizing:** Beam width is dynamically adjusted based on input sequence density to optimize speed without sacrificing quality.

### 4. Postprocessing & Cleanup
The `VectorizedPostprocessor` refines the English output by:
* **Repetition Removal:** Detecting and collapsing multi-word repeating phrases (a common ByT5 artifact).
* **Historical Terminology:** Converting literal translations into correct historical terms (e.g., `shekels less grains`).
* **Character Filtering:** Safely stripping forbidden symbols while protecting semantic brackets and `<gap>` tags.

---

## Performance & Evolution
* **v13 (34.3):** Single-model, composite MBR
* **v16 (34.8):** Two-model, pure chrF++ MBR
* **v17 (35.4):** Two-model, geo-mean(BLEU × chrF++) MBR
* **v19 (35.5):** Pipeline + standard agreement bonus
* **v20 (35.6):** Geo-mean, separate pools, cross-model bonus, ε×6
* **v39 (35.4):** Composite, separate pools, cross-model bonus, ε×6
* **v40 (35.6):** Composite, separate pools, cross-model bonus, temps [0.55,0.75,0.95], ε×6
* **v41 (36.2):** Composite, MERGED pools, NO cross-model bonus, temps [0.55,0.75,0.95], ε×6

---

## Requirements & Environment
* **Hardware:** Optimized for NVIDIA Tesla T4 (Kaggle environment).
* **Precision:** Automatic Mixed Precision (**BF16**) for inference speed.
* **Key Libraries:** `transformers`, `torch`, `sacrebleu`, `pandas`, `optimum` (BetterTransformer).

---
## Project Link
For the full implementation and interactive notebook, visit the Kaggle project page:
**[Kaggle: Hybrid MBR Ensemble for Akkadian Translation](https://www.kaggle.com/code/francescocampigotto/hybrid-mbr-ensemble-for-akkadian-translation)**
