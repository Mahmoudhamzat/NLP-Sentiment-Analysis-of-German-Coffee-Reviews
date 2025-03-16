# NLP-Sentiment-Analysis-of-German-Coffee-Reviews
```markdown
# Coffee Review Sentiment Classification Project

Welcome to the Coffee Review Sentiment Classification Project! This project aims to analyze the sentiment of coffee reviews using natural language processing (NLP) techniques and machine learning. It relies on Python libraries such as `pandas`, `NumPy`, and `scikit-learn`, along with `spaCy` for text processing.

## Table of Contents
- [Description](#description)
- [Requirements](#requirements)
- [Files](#files)
- [Installation](#installation)
- [Usage](#usage)
- [How the Project Works](#how-the-project-works)
- [Contributing](#contributing)
- [License](#license)

## Description

This project analyzes coffee reviews to determine whether the review is positive or negative based on the user's rating. It uses machine learning methods like Gaussian Naive Bayes to classify the sentiments.

## Requirements

Before you begin, ensure you have the following requirements installed:

- Python 3.x
- Python libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `spacy`
  - `joblib`

You can install the required libraries using:

```bash
pip install numpy pandas scikit-learn spacy joblib
python -m spacy download de_core_news_sm
```
## Files

- `kaffee_reviews.csv`: A file containing coffee reviews.
- `sentiment_classifier_count.pkl`: Sentiment classification model using CountVectorizer.
- `sentiment_classifier_tfidf.pkl`: Sentiment classification model using TF-IDF.
- `script_name.py`: The source code for data analysis.

## Installation

1. Clone or download this repository to your local machine:
   ```bash
   git clone https://github.com/username/repository.git
   cd repository
   ```

2. Make sure to install all the required libraries as outlined above.

## Usage

1. Ensure that the `kaffee_reviews.csv` file is located in the specified path in the code.
2. Run the program using Python:
   ```bash
   python script_name.py
   ```

3. The confusion matrix and classification reports will be printed to the console, along with a display of the cleaned data with sentiment labels.

## How the Project Works

- **Data Loading**: Loads data from a CSV file.
- **Data Processing**: Cleans and processes text using techniques such as stop word removal and lemmatization.
- **Feature Extraction**: Uses CountVectorizer and TF-IDF to extract features from the text.
- **Model Building**: Constructs a sentiment classification model using Gaussian Naive Bayes.
- **Performance Evaluation**: Evaluates the model's performance using a confusion matrix and classification reports.

## Contributing

If you would like to contribute to this project, please feel free to open a pull request or create an issue to discuss your ideas for improvements.


