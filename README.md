# Fake Product Review Detection Using Machine Learning and Sentiment Analysis

[![Python Version](https://img.shields.io/badge/Python-3.13.7-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Unit%20Tests-25%2F25%20Passing-brightgreen.svg)]()

> **B.Tech CSE (Artificial Intelligence & Machine Learning) — Academic Mini-Project**  
> An explainable, leak-free, classical Natural Language Processing and Machine Learning system for detecting incentivized and fraudulent e-commerce reviews using multi-modal stylometric, sentiment, and n-gram representations.

---

## 1. Executive Summary & Verified Experimental Results

Unlike naive text classifiers that rely solely on word frequency or opaque deep learning transformers, this system combines **negation-preserving TF-IDF vectorization**, **linguistic stylometrics** (length, capitalization, punctuation density, lexical richness), and **lexical sentiment polarity** (NLTK VADER) into a unified, mathematically auditable pipeline.

### Consolidated Model Comparison Benchmark
Evaluated on **$6,595$ held-out test reviews** using **GroupShuffleSplit on `reviewer_id`** ($0\%$ author leakage):

| Pipeline | Model Architecture | Accuracy | Macro F1 | Fake Precision | Fake Recall | Honest Recall | ROC-AUC | PR-AUC | Train Time | Inference Latency | Model Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Enhanced (B)** | **Logistic Regression (Balanced)** | **78.30%** | **75.92%** | **85.98%** | **81.14%** | **72.37%** | **0.8439** | **0.9063** | 26.05s | 65.26 ms / 100 | 1.15 MB |
| **Enhanced (B)** | SGD Log-Loss Classifier | 77.68% | 75.28% | 85.66% | 80.47% | 71.85% | 0.8358 | 0.9008 | 19.48s | 64.54 ms / 100 | 1.15 MB |
| **Enhanced (B)** | Linear SVM (Balanced) | 76.29% | 73.31% | 83.38% | 81.10% | 66.23% | 0.8188 | 0.8911 | 45.30s | 64.13 ms / 100 | 1.15 MB |
| **Baseline (A)** | Logistic Regression (Historical) | 76.74% | 74.06% | 84.39% | 80.49% | 68.90% | 0.8186 | 0.8888 | 1.75s | 3.32 ms / 100 | 0.92 MB |
| **Baseline (A)** | SGD Log-Loss Classifier | 76.39% | 73.83% | 84.57% | 79.62% | 69.65% | 0.8116 | 0.8820 | 1.75s | 3.35 ms / 100 | 0.92 MB |
| **Baseline (A)** | Linear SVM (Historical) | 74.75% | 71.52% | 82.07% | 80.18% | 63.42% | 0.8003 | 0.8811 | 1.74s | 3.31 ms / 100 | 0.92 MB |
| **Baseline (A)** | Random Forest (Depth 25) | 73.01% | 70.50% | 83.02% | 75.54% | 67.73% | 0.7856 | 0.8658 | 2.09s | 3.97 ms / 100 | 2.64 MB |
| **Baseline (A)** | Multinomial Naive Bayes (Smoothed) | 72.86% | 58.18% | 72.10% | 97.67% | 21.03% | 0.8035 | 0.8838 | 1.60s | 3.32 ms / 100 | 1.80 MB |

### Feature Ablation Impact (Pipeline A vs. Pipeline B)
Direct comparison on identical test folds proves that adding stylometrics and VADER sentiment produces an empirical lift across every metric:
- **Macro F1:** $+1.86\%$ lift ($74.06\% \rightarrow 75.92\%$)
- **Accuracy:** $+1.56\%$ lift ($76.74\% \rightarrow 78.30\%$)
- **Honest Review Recall:** $+3.47\%$ lift ($68.90\% \rightarrow 72.37\%$)
- **ROC-AUC:** $+0.0253$ lift ($0.8186 \rightarrow 0.8439$)

---

## 2. Project Architecture & Pipeline Flow

```mermaid
flowchart TD
    subgraph Data Layer
        D1["Primary Dataset (He et al., 2022)<br/>32,940 Ground-Truth Reviews"]
        D2["Secondary Benchmark (Salminen et al.)<br/>40,432 AI Reviews (10 Categories)"]
        D3["Unlabeled Corpus (Stanford Fine Foods)<br/>568,454 Unsupervised Baseline"]
    end

    subgraph Preprocessing & Leakage Control
        P1["Unified 12-Column Schema Loader"]
        P2["GroupShuffleSplit by Reviewer ID<br/>(0% Author Leakage)"]
        P3["Text Normalizer<br/>(Contractions, HTML, URLs, Negations)"]
    end

    subgraph Feature Engineering (FeatureUnion)
        F1["Word TF-IDF (1-2 grams, Negation-Preserved)"]
        F2["Character TF-IDF (2-5 sub-word grams)"]
        F3["Stylometric Profiler (Length, CAPS, Exclamations, TTR)"]
        F4["VADER Sentiment Engine (Pos, Neg, Neu, Compound)"]
    end

    subgraph Machine Learning & Explainability
        M1["Regularized Logistic Regression (Balanced)"]
        M2["Linear Support Vector Machine (SVC)"]
        M3["Multinomial Naive Bayes (Smoothed)"]
        M4["Random Forest & SGD Classifiers"]
        X1["Log-Odds Feature Attribution (XAI)"]
        X2["Cross-Domain Transfer Analysis"]
    end

    subgraph Presentation Layer
        APP["Streamlit Multi-Page Web App (8 Dashboards)"]
    end

    Data Layer --> Preprocessing & Leakage Control
    Preprocessing & Leakage Control --> Feature Engineering
    Feature Engineering --> Machine Learning & Explainability
    Machine Learning & Explainability --> Presentation Layer
```

---

## 3. Dataset Provenance & Integrity

### Primary Dataset: Sherry He et al. (2022) — *Direct Observational Ground Truth*
- **Citation:** He, S., Hollenbeck, B., & Proserpio, G. (2022). *The Market for Fake Reviews.* Marketing Science / NBER Working Paper No. 27671.
- **Ground Truth Basis:** Direct empirical tracking of underground Facebook review-recruitment groups where Amazon third-party sellers paid consumers for 5-star product reviews via PayPal refunds.
  - `reviewer_labeled_fake = 1`: Caught directly inside incentivized groups.
  - `reviewer_labeled_honest = 1`: Control reviewers with no illicit presence.
- **Scale:** $32,940$ reviews ($22,138$ Fake [67.21%], $10,802$ Genuine [32.79%]).

### Secondary Dataset: Joni Salminen et al. (2022) — *Cross-Dataset Generalization*
- **Citation:** Salminen, J., et al. (2022). *Creating and detecting fake reviews of extreme-sentiment.* Springer Nature / Electronic Commerce Research.
- **Scale:** $40,432$ reviews across 10 Amazon categories ($20,216$ CG [Computer Generated via GPT-2], $20,216$ OR [Original Human]).

### Unlabeled Dataset: Stanford Amazon Fine Food Reviews
- **Scale:** $568,454$ reviews.
- **Integrity Notice:** Documented as having **zero deception labels**. Kept strictly as an out-of-domain unlabeled corpus for unsupervised sentiment testing; never used for supervised training.

---

## 4. Key Engineering Innovations & Methodologies

1. **Reviewer-Group Leakage Prevention (`GroupShuffleSplit`):**
   Reviews from the same author share idiosyncratic phrasing and vocabulary. Random train/test splits allow reviewer style to leak into test sets. Splitting by `reviewer_id` guarantees zero author overlap between training ($9,386$ authors) and testing ($2,347$ authors).
2. **Negation-Preserving NLP:**
   Standard stopword lists purge `"not"`, `"never"`, `"no"`, inverting sentiment in reviews like *"not good"*. Our custom pipeline retains all negation tokens and expands contractions (`"don't"` $\rightarrow$ `"do not"`).
3. **Multi-Modal Feature Union:**
   Combines vocabulary n-grams with 9 stylometric markers (character count, word count, avg word length, exclamation count, question count, uppercase ratio, punctuation ratio, 1st-person pronouns, lexical diversity) and 4 standardized VADER continuous scores.
4. **Cross-Dataset Generalization (Negative Transfer Discovery):**
   Evaluating the model trained on human fraud against AI-generated reviews revealed a transfer drop from $78.30\%$ to $52.11\%$. This provides critical academic evidence that **models trained on human financial incentive fraud do not automatically generalize to AI-generated text without domain adaptation**.

---

## 5. How to Run the Project (Step-by-Step on Windows 11)

### Prerequisites
- Python 3.10+ (Tested on Python 3.13.7)
- Google Chrome or any modern web browser

### Step 1: Clone the Repository
```powershell
git clone https://github.com/MLxTRizvi/initial-project.git
cd "mini project 2"
```

### Step 2: Set Up Virtual Environment & Dependencies
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### Step 3: Run the Automated Unit Test Suite (25 Tests)
```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

### Step 4: Launch the Streamlit Web Application
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Streamlit will automatically open your default browser (Google Chrome) at:
```
http://localhost:8501
```

*(If Chrome does not open automatically, simply copy `http://localhost:8501` into Chrome's address bar).*

---

## 6. Project Directory Architecture

```
mini project 2/
├── app.py                             # Full-featured 8-page Streamlit Analytics Application
├── app_original_baseline.py           # Historical baseline app (preserved for reproducibility)
├── requirements.txt                   # Production package dependencies
├── README.md                          # Academic and technical documentation
├── .gitignore                         # Sanitized Git ignore rules (protects large files)
│
├── src/                               # Modular Python Source Package
│   ├── __init__.py
│   ├── data_loader.py                 # Unified 12-column schema ingestion & validators
│   ├── preprocessing.py              # Text cleaning, contractions, negation preservation
│   ├── feature_engineering.py        # FeatureUnion: Word/Char TF-IDF + Stylistics + VADER
│   ├── sentiment_analysis.py          # NLTK VADER analyzer & scikit-learn transformer
│   ├── evaluate.py                    # Multi-metric evaluator & latency benchmarking
│   ├── train.py                       # Leak-free training, cross-validation, ablation runner
│   ├── explainability.py              # XAI global log-odds & local word attributions
│   └── cross_dataset.py               # Transfer evaluation on 40,432 Salminen reviews
│
├── tests/                             # Automated Unit Test Suite (25 Tests)
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_sentiment_analysis.py
│   ├── test_feature_engineering.py
│   ├── test_model_pipeline.py
│   └── test_explainability.py
│
├── models/                            # Serialized Model Artifacts
│   ├── best_model_enhanced.joblib     # Enhanced Multi-Modal Classifier (1.15 MB)
│   └── best_model.joblib              # Historical Baseline Classifier (2.44 MB)
│
├── results/
│   ├── figures/                       # 13 High-Resolution Figures (300 DPI)
│   │   ├── 01_class_distribution_comparison.png
│   │   ├── 02_review_length_distribution.png
│   │   ├── 03_review_length_boxplots.png
│   │   ├── 04_rating_distribution_by_label.png
│   │   ├── 05_top_words_genuine_vs_fake.png
│   │   ├── 06_category_distribution_by_label.png
│   │   ├── 07_linguistic_markers_comparison.png
│   │   ├── 08_sentiment_distribution_by_class.png
│   │   ├── 09_roc_pr_curves.png
│   │   ├── 10_confusion_matrices_comparison.png
│   │   ├── 11_threshold_tuning_curve.png
│   │   ├── 12_top_feature_coefficients.png
│   │   └── 13_cross_domain_category_performance.png
│   └── metrics/                       # Raw Empirical Benchmark CSVs & JSONs
│       ├── eda_summary.json
│       ├── model_comparison_comprehensive.csv
│       ├── feature_ablation_comparison.csv
│       ├── threshold_tuning.csv
│       ├── top_feature_coefficients.csv
│       ├── cross_dataset_transfer.csv
│       └── cross_dataset_category_performance.csv
│
├── notebooks/
│   ├── 01_comprehensive_eda.ipynb     # Step-by-step academic exploration notebook
│   └── EDA.ipynb                      # Historical EDA notebook (preserved)
│
└── docs/
    └── dataset_benchmarks_and_expansion.md  # Survey of YelpChi, YelpNYC, OpSpam, etc.
```

---

## 7. Critical Academic Viva Defense Q&A

### Q1: "Why did you use Classical Machine Learning instead of BERT or Transformers?"
> *"Our project prioritizes auditability, inference speed, and resistance to domain overfitting. Linear models with regularized multi-modal features achieve ~78.3% accuracy with an inference latency of 65 ms per 100 samples and a footprint of only 1.15 MB. Crucially, linear models yield exact mathematical log-odds coefficients ($e^{w_i}$), allowing us to explain to an e-commerce compliance auditor exactly which lexical and stylometric tokens triggered a suspicion flag—something black-box transformers cannot do without heavy approximations."*

### Q2: "Why did you use GroupShuffleSplit instead of standard train_test_split?"
> *"Random splitting causes severe reviewer-level data leakage. In review datasets, prolific authors contribute multiple reviews with unique stylistic habits. If reviews from the same author appear in both training and test folds, the classifier learns author identification rather than generalizable signals of deception. Group splitting by `reviewer_id` guarantees zero author overlap between folds."*

### Q3: "What did your Feature Ablation experiment prove?"
> *"Under identical data splits, adding character n-grams, stylometric markers (length, uppercase ratio, exclamations, pronoun frequency, lexical diversity), and VADER sentiment scores provided a consistent empirical lift across every metric: Accuracy increased from 76.74% to 78.30% (+1.56%), Macro F1 from 74.06% to 75.92% (+1.86%), and Honest Review Recall from 68.90% to 72.37% (+3.47%)."*

### Q4: "Why did your model experience a performance drop on the Salminen dataset?"
> *"This reflects negative transfer due to the difference in the underlying generative mechanism. Our primary model was trained on human financial incentive fraud (recruited via Facebook), where reviewers use excessive length and emotional rhetoric to get refunds. In contrast, the Salminen dataset contains machine-generated text (GPT-2). The model correctly proves that human fraud patterns do not automatically transfer to synthetic AI reviews without domain adaptation."*

---

## 8. Academic References

1. **He, S., Hollenbeck, B., & Proserpio, G. (2022).** *The Market for Fake Reviews.* Marketing Science / NBER Working Paper No. 27671.
2. **Salminen, J., Kandpal, C., Kamel, A. M., Jung, S. G., & Jansen, B. J. (2022).** *Creating and detecting fake reviews of extreme-sentiment: an interactive and transparent approach.* Springer Nature / Electronic Commerce Research.
3. **Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011).** *Finding Deceptive Opinion Spam by Any Means Necessary.* In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011), pp. 309–319.
4. **Hutto, C., & Gilbert, E. (2014).** *VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text.* In Proceedings of the International AAAI Conference on Web and Social Media (ICWSM 2014).
5. **Rayana, S., & Akoglu, L. (2015).** *Collective Opinion Spam Detection: Bridging Review Networks and Metadata.* In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2015), pp. 985–994.

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
