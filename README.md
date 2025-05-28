# Fake-news-detection
📰 Fake News Detection with Machine Learning – Classifying News Authenticity Using NLP and Ensemble Models

# 📰 Fake News Detection Using Machine Learning

A machine learning project to classify news articles as real or fake using NLP and ensemble models. Built using a Kaggle dataset and powered by multiple classification algorithms including Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.

---

## 📘 Extended Description

In an era of digital misinformation, this project addresses the challenge of identifying fake news by analyzing a dataset of real and fake news articles. Using Natural Language Processing (NLP) techniques and a series of machine learning models, the project classifies articles with high accuracy and provides tools for both manual and automated evaluation.

The solution involves comprehensive preprocessing (text cleaning, tokenization, vectorization), feature engineering using TF-IDF, and model training/testing using Scikit-learn classifiers. The end product is a fake news classifier with built-in performance evaluation and visualization capabilities, including ROC curves and classification reports.

This project not only demonstrates strong model accuracy (up to 99.5%) but also showcases how ensemble models and clean data pipelines can be used to combat misinformation at scale.

---


---

## 🚀 Features

- 🧹 Clean and preprocess raw news text using regex and string methods
- 🔍 Feature extraction using TF-IDF vectorization
- ⚙️ Train and compare multiple models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- 📉 Evaluate models using:
  - Accuracy
  - Classification report (Precision, Recall, F1-Score)
  - Confusion matrix
  - ROC Curve and AUC
- 🧪 Manual testing tool for live article classification

---

## 🛠️ Tools & Technologies

| Category           | Tools / Libraries                               |
|-------------------|--------------------------------------------------|
| Language           | Python                                           |
| Data Handling      | pandas, numpy                                    |
| Visualization      | matplotlib, seaborn                              |
| Text Processing    | re, string                                       |
| NLP & ML           | scikit-learn, TfidfVectorizer                    |
| Models             | LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier |

---

## 📊 Model Performance Highlights

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 98.7%    |
| Decision Tree       | 99.5%    |
| Gradient Boosting   | 99.5%    |
| Random Forest       | 98.8%    |

---

## 🔍 Sample Manual Testing Function

```python
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_xv_test = vectorization.transform(new_def_test["text"])
    
    print("LR Prediction:", output_lable(LR.predict(new_xv_test)[0]))
    print("DT Prediction:", output_lable(DT.predict(new_xv_test)[0]))
    print("GBC Prediction:", output_lable(GBC.predict(new_xv_test)[0]))
    print("RFC Prediction:", output_lable(RFC.predict(new_xv_test)[0]))

