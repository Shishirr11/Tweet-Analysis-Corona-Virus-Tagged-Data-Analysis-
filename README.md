## Introduction

This project analyzes COVID-19 related tweets and predicts the sentiment of each tweet using two transformer models: **BERT** and **RoBERTa**. The core workflow is:

1) load the tagged tweet dataset  
2) clean and normalize tweet text  
3) reduce the original 5-level sentiment into **3 classes** (Negative / Neutral / Positive)  
4) balance the training set using oversampling  
5) train and evaluate **BERT** and **RoBERTa** models side-by-side

The implementations in a single notebook: `Tweet Analysis using BERT and RoBERTa.ipynb`.

---

## Data description

The dataset is included in the repo under:

- `Tweet Data For Corona Virus/Corona_NLP_train.csv` (41,157 rows)
- `Tweet Data For Corona Virus/Corona_NLP_test.csv` (3,798 rows)

Each file contains 6 columns:

- `UserName`
- `ScreenName`
- `Location`
- `TweetAt`
- `OriginalTweet` (raw tweet text)
- `Sentiment` (original labels)

### Label setup used in this project (5 → 3)
The notebook collapses the original sentiment labels into 3 classes:

- **0 = Negative**: `Extremely Negative`, `Negative`
- **1 = Neutral**: `Neutral`
- **2 = Positive**: `Positive`, `Extremely Positive`

---

## Features and output

### Text cleaning (applied to train + test)
The notebook creates a cleaned column (`text_clean`) from `OriginalTweet` using:

- emoji stripping
- removing URLs and @mentions
- removing non-ASCII characters
- removing punctuation and a few common “broken encoding” characters
- cleaning hashtag tokens (splitting `#something_like_this` into readable words)
- dropping tokens that contain `$` or `&`

### Class balancing (train only)
The training set is balanced using **RandomOverSampler**, so the model doesn’t get biased toward the majority class.

### Model training (two runs)
The notebook trains and evaluates two models:

- **BERT**: `bert-base-uncased`
- **RoBERTa**: `roberta-base`

Common settings used:
- max sequence length: **128**
- epochs: **4**
- batch size: **32**
- optimizer: Adam (lr = 1e-5)
- output: 3-class softmax

### Outputs you get when you run the notebook
- training/validation learning curves (from `model.fit` history)
- **confusion matrix** plots
- **classification reports** (precision/recall/F1) for:
  - BERT
  - RoBERTa

---

## Prerequisites

- Python 3.8+
- Jupyter Notebook / JupyterLab **or** Google Colab
- Required libraries (the notebook imports these):
  - `pandas`, `numpy`
  - `scikit-learn`
  - `tensorflow`
  - `transformers`
  - `imblearn`
  - `emoji`
  - `matplotlib`, `seaborn`
  - (optional but used) `nltk`

If you’re running locally, make sure TensorFlow is installed.

---

## How to run it

### Jupyter (local)
1. Unzip / clone the project folder.
2. Open the notebook:
   - `Tweet Analysis using BERT and RoBERTa.ipynb`
3. Make sure the CSV paths match your folder.
   - The notebook reads `Corona_NLP_train.csv` and `Corona_NLP_test.csv`.
   - If needed, update the paths to point to:
     - `Tweet Data For Corona Virus/Corona_NLP_train.csv`
     - `Tweet Data For Corona Virus/Corona_NLP_test.csv`
4. Run cells top-to-bottom:
   - first section trains/evaluates **BERT**
   - later section repeats training/evaluation for **RoBERTa**

### Google Colab
1. Upload the notebook and the `Tweet Data For Corona Virus/` folder to Colab (or mount Drive).
2. Update the `pd.read_csv(...)` paths if necessary.
3. Run all cells in order.

That’s it, you’ll see the confusion matrices and classification reports for both models.
