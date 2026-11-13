
# Fake Online Review Detection Using NLP and Machine Learning

## 1. Project Overview

Fake Online Review Detection is a machine learning project that classifies online product reviews into two categories:

- **Fake-labelled Review (1)**
- **Honest-labelled Review (0)**

The project uses Natural Language Processing (NLP) and machine learning algorithms to analyse review text and identify patterns associated with the labels in the dataset.

**Note:** The predictions reflect the dataset's reviewer annotations. They do not independently prove whether a review is fraudulent.

## 2. Objectives

- Analyse and preprocess an online review dataset.
- Convert textual reviews into numerical features using TF-IDF.
- Train and compare multiple machine learning algorithms.
- Evaluate model performance using classification metrics.
- Develop a simple web application for review classification.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)
- Streamlit
- Joblib
- Jupyter Notebook

## 4. Dataset Description

The project uses a public online product review dataset.

The original dataset contains 381,734 records and 23 columns.

After preprocessing and selecting records with unambiguous reviewer labels, the final dataset contains **32,940 reviews**.

| Label | Category | Number of Reviews |
|---|---|---:|
| 0 | Honest-labelled | 10,802 |
| 1 | Fake-labelled | 22,138 |
| **Total** | | **32,940** |

The target labels are derived from the `reviewer_labeled_fake` and `reviewer_labeled_honest` columns.

## 5. Project Workflow

1. Dataset collection
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Text preprocessing and TF-IDF feature extraction
5. Model training
6. Model comparison and evaluation
7. Saving the best model
8. Building a Streamlit web application

## 6. Machine Learning Algorithms

Three classification algorithms were implemented:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (SVM)

The models use TF-IDF features extracted from the review title and review text.

A reviewer-grouped train-test split was used to prevent reviews from the same reviewer appearing in both training and testing sets.

## 7. Model Performance

The models were evaluated using accuracy, precision, recall, F1-score, and macro F1-score.

| Algorithm | Accuracy | Macro F1 |
|---|---:|---:|
| Naive Bayes | 70.13% | 49.34% |
| Logistic Regression | 76.62% | 74.02% |
| Linear SVM | 74.91% | 71.67% |

**Best Performing Model: Logistic Regression**

Logistic Regression achieved the highest macro F1-score of approximately **74.02%** and was selected as the final model.

## 8. Web Application

A simple interactive web application was developed using Streamlit.

Users can enter a review title and review text. The application uses the trained model to predict whether the review is fake-labelled or honest-labelled.

## 9. How to Run the Project

### Step 1: Clone the repository

```bash
git clone <your-repository-url>
cd Fake-Review-Detection
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Prepare the dataset

Place the original dataset inside the `dataset` folder with the filename:

`public_reviews_dataset_cleaned.csv`

### Step 6: Run preprocessing

```bash
python preprocessing.py
```

### Step 7: Train the models

```bash
python train_model.py
```

### Step 8: Launch the web application

```bash
streamlit run app.py
```

## 10. Project Structure

```text
Fake-Review-Detection/
│
├── dataset/
├── models/
├── outputs/
├── notebooks/
│   └── EDA.ipynb
│
├── preprocessing.py
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 11. Limitations

- Model performance depends on the quality and reliability of the dataset annotations.
- The dataset labels represent reviewer-level annotations and may not establish the authenticity of every individual review.
- Text-based classification cannot guarantee the detection of all fake reviews.
- The current project uses traditional machine learning algorithms rather than deep learning.

## 12. Future Scope

- Implement deep learning models such as LSTM or BERT.
- Explore additional features such as reviewer behaviour and rating patterns.
- Improve model performance using hyperparameter tuning.
- Evaluate the model on independently verified review datasets.
- Deploy the application as an online service.

## 13. Conclusion

This project demonstrates how Natural Language Processing and machine learning can be applied to online review classification. Three classification algorithms were compared, and Logistic Regression achieved the best macro F1-score of approximately 74.02%.

The project also provides an interactive Streamlit application for testing review text.
