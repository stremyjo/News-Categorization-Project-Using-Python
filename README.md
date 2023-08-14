# News Categorization Project

## Overview

This project focuses on categorizing news articles into different categories using natural language processing (NLP) techniques and machine learning. The main steps of the project include data preprocessing, feature extraction, model training, and evaluation.

## Getting Started

1. **Data Preparation**: We start by reading the news articles data from a CSV file, handling duplicates, converting to lowercase, and dealing with missing values.

2. **Tokenization**: We tokenize the headlines in the news articles to break them into individual words.

3. **Stopword Removal**: Common stopwords (e.g., "the," "and," "is") are removed from the token lists to improve data quality.

4. **Stemming**: We apply stemming to further simplify the words by reducing them to their root forms.

5. **Special Character Removal**: We remove special characters from the token lists using regular expressions.

## Model Building

1. **Category Definition**: We define categories and their associated keywords, which guide our categorization process.

2. **Categorization Function**: We create a function to categorize tokens based on the defined categories and keywords. This function helps assign a category to each article.

3. **Model Training**: We use the Multinomial Naive Bayes algorithm to train the categorization model.

4. **Model Evaluation**: We evaluate the model using classification metrics such as accuracy and the `classification_report`.

## Usage

To use the trained model on new news articles:

1. Read the new news articles CSV file.
2. Perform the same preprocessing steps as described above.
3. Convert the processed token lists to TF-IDF numerical features.
4. Use the trained model to make predictions on the new data.

## Example

You can find an example of how to use the trained model on new news articles in the second part of the code. The code loads the trained model, reads new news articles from a CSV file, applies preprocessing, converts text data to TF-IDF features, makes predictions, and prints the results.

## Conclusion

This project demonstrates the process of categorizing news articles using NLP techniques and a machine learning model. The trained model can be applied to new news articles to predict their categories, making it a useful tool for automated news categorization.
