# House Price Prediction 🏠

A machine learning project that predicts house sale prices based on property features, using the [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) from Kaggle.

## Project Overview

This project walks through a complete data science workflow:
1. Data exploration and cleaning
2. Handling missing values
3. Outlier detection and removal
4. Feature engineering (One-Hot Encoding for categorical data)
5. Model training and comparison

## Dataset

The dataset contains 1,460 houses with 79 explanatory variables describing (almost) every aspect of residential homes, including size, quality, year built, and neighborhood.

## Data Cleaning

- Dropped columns with excessive missing values (`PoolQC`, MiscFeature, Alley, `Fence`)
- Filled missing numerical values (`LotFrontage`) with the median
- Filled missing categorical values (e.g., GarageType, BsmtQual, FireplaceQu`) with `'None', since missing data represented the absence of a feature (e.g., no garage) rather than an error
- Removed two outliers with unusually large living area but low sale price

## Feature Engineering

- Applied One-Hot Encoding to the Neighborhood column, converting it into 24 binary features
- Final feature set includes: GrLivArea, OverallQual, YearBuilt, TotalBsmtSF, GarageCars, and all encoded Neighborhood columns

## Models & Results

| Model | Features | Mean Absolute Error |
|---|---|---|
| Linear Regression | GrLivArea only | $40,086 |
| Linear Regression | 5 numerical features | $24,530 |
| Linear Regression | 5 features + Neighborhood | $21,686 |
| Random Forest | 5 features + Neighborhood | $19,149 |

Random Forest outperformed Linear Regression, likely due to its ability to capture non-linear relationships between features and price.

## Tools Used

- Python (pandas, matplotlib, scikit-learn)
- Google Colab

## Next Steps

- Add more features (e.g., FullBath, `YearRemodAdd`)
- Try Gradient Boosting models (e.g., XGBoost)
- Hyperparameter tuning for Random Forest
- Generate predictions on the Kaggle test set for submission

---
## German Sentiment Analysis 🇩🇪

A fine-tuned BERT model that classifies German text reviews as positive or negative, built using Hugging Face Transformers.

### Project Overview

This project fine-tunes bert-base-german-cased on a German subset of a multilingual Amazon reviews dataset to perform binary sentiment classification.

Workflow:
1. Loading and filtering a multilingual dataset for German-language reviews
2. Tokenizing text using a German BERT tokenizer
3. Fine-tuning bert-base-german-cased for sequence classification
4. Evaluating model performance on a held-out test set

### Results

- Test Accuracy: 93.95%
- Test Loss: 0.215

### Tech Stack

- Python, PyTorch
- Hugging Face transformers & datasets
- Google Colab (GPU-accelerated training)

### Model

The fine-tuned model is available in the [`my_model`](./my_model) directory and can be loaded directly with:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
## Author

Maryam Noroozi
