# Fake News Detection Using PySpark & Machine Learning

This project detects fake news using a combination of **PySpark** for big data preprocessing and **scikit-learn** for modeling. It uses the `FA-KES-Dataset.csv` dataset, which includes real and fake news articles with metadata like source, date, location, and labels.

## Dataset Overview

- **Source**: `FA-KES-Dataset.csv`
- **Columns**:
  - `unit_id`: Unique identifier
  - `article_title`: News headline
  - `article_content`: Full news content
  - `source`: Origin of the news
  - `date`: Published date
  - `location`: Location related to news
  - `labels`: 1 = Real News, 0 = Fake News

## Technologies Used

- Apache Spark -> PySpark
- TextBlob -> for sentiment analysis
- scikit-learn
- imbalanced-learn (SMOTE)
- Seaborn, Matplotlib (visualization)
- WordCloud (visuals)

## Pipeline

### 1. Data Cleaning & Preprocessing (PySpark)
- Dropped nulls from `article_title`, `article_content`, `labels`
- Cleaned text using regex and lowercasing
- Tokenized and removed stopwords

### 2. Feature Engineering
- TF-IDF vectorization
- Sentiment Analysis using TextBlob (`Title_Sentiment`)
- Added `Word_Count` and `Char_Count` features

### 3. Visualization
- Word clouds for fake and real articles
- Histograms for word count and sentiment scores
- Label distribution plots

### 4. Handling Imbalanced Data
- Applied **SMOTE** on extracted features (`Word_Count`, `Char_Count`, `Title_Sentiment`)


## Models & Evaluation

### Models Used
- **Logistic Regression**
- **Naive Bayes (GaussianNB)**
- **Random Forest Classifier**

### Metrics Evaluated
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix

### Hyperparameter Tuning
- **GridSearchCV** used for:
  - Gaussian Naive Bayes
  - Random Forest


## Results Summary

| Model                   | Accuracy |
|-------------------------|----------|
| Logistic Regression     | ~86%     |
| Naive Bayes (Tuned)     | ~85%     |
| Random Forest (Tuned)   | ~89%     |

> Visual comparison of accuracy and performance metrics provided in bar charts and heatmaps.

## Key Insights

- **Title sentiment** is a strong indicator of fake news bias.
- Real news tends to have higher word counts.
- Class balancing with SMOTE improved model generalization.


## Future Improvements

- Integrate deep learning (LSTM or BERT) for semantic context.
- Expand feature space with n-grams and readability scores.
- Deploy on a web interface for real-time article classification.

