# Beginner-Friendly Python Text Cleaning Project

## Text Classification for News Articles Project

Are you excited to learn how to clean and process text data using Python? In this beginner-friendly project, we'll walk through the steps of cleaning news data to make it more usable for analysis. Don't worry if you're new to programming – we'll take it step by step!

I chose to run this code using ‘Jupyter Lab’, you can launch it by going to your command program and navigating to the default library where python is installed on your computer. If python is installed to your path, you do not need to do this and can simply type:

```bash
jupyter notebook
```

Click enter, which will launch Jupyter notebook in your web browser. You can add whichever relevant data files you need by dragging the files to the directory from your finder or file explorer window and following any on-screen prompts that appear.

We will use these files here for this project: [https://archive.ics.uci.edu/dataset/359/news+aggregator](https://archive.ics.uci.edu/dataset/359/news+aggregator)

## Step 1: Download the News Aggregator Dataset

### [Download the News Aggregator Dataset](https://archive.ics.uci.edu/dataset/359/news+aggregator)

You will have to extract the file and drag the ‘newsCorpora’ and ‘2pageSessions’ files into your Jupyter lab by dragging them into your files tab in the Jupyter lab web page as described above.

**Dataset Details:**

Dataset of references (urls) to news web pages

Dataset of references to news web pages collected from an online aggregator in the period from March 10 to August 10 of 2014. The resources are grouped into clusters that represent pages discussing the same news story. The dataset includes also references to web pages that point (has a link to) one of the news page in the collection.

**TAGS:** web pages, news, aggregator, classification, clustering

## Step 2: Launch a new Jupyter Notebook

Go to File and then ‘New’ and select ‘Notebook’. You will see an empty bar where you can begin writing your code. Choose your default python kernel.

## Step 3: Download and Import Required Packages

Begin by importing necessary packages, you can copy this and paste it into the cell and run it by pressing ‘shift’ and ‘enter’ on your keyboard at the same time or pressing run in the top bar.

```python
import pandas as pd          # Data manipulation
import nltk                 # Natural Language Processing (NLP)
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split   # For splitting the data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer   # Feature extraction
from sklearn.naive_bayes import MultinomialNB          # Naive Bayes classifier
from sklearn.metrics import classification_report, accuracy_score   # Model evaluation
```

Go to your command window to install these packages if you don’t have them using this code for each missing package:

```bash
pip install scikit-learn
```

```bash
pip install nltk
```

```bash
pip install pandas
```

## Step 4: Load the dataset into a pandas DataFrame

Use the following code to read the first csv file:

```python
import pandas as pd

# Read the CSV file into a DataFrame, skipping lines with parsing errors
data = pd.read_csv('[insert full path to your newsCorpora.csv file here]', on_bad_lines='skip')

# Now you can work with the 'data' DataFrame
```

When using the full path name, you can use a double backslash ('\\') to prevent unicode escape errors in the file path. This happens because the backslash ('\\') character is treated as an escape character in Python strings.

If you peeked and looked at the dataset, you will also notice that the table is structured differently than we’re used to big data files looking, no multiple columns, and only thousands of rows with the links. This can cause us issues in Python using our code as is. So you can alternatively use the following code when importing/reading the csv file in python by indicating that there are no headers.

```python
import pandas as pd

# Read the CSV file into a DataFrame, skipping lines with parsing errors
data = pd.read_csv('path_to_your_data.csv', on_bad_lines='skip', header=None, usecols=[0])

# Now you can work with the 'data' DataFrame
```

Alternatively, you can define column names based on the attributes of the news articles, and then read the CSV file and specify the column names.

```python
column_names = ['headline', 'attribute1', 'attribute2', ...]  # Replace with actual attribute names
```

```python
data = pd.read_csv('path_to_your_data.csv', names=column_names, header=None)
```

Because we only have one column that we’re working with, I opted for using no headers or column names, the rest of my code reflects this but you can change the data’[0]’ portion of my codes to reflect your column names or identifiers. For example, if your column was named ‘headline’ as per this code: 

```python
column_names = ['headline', 'attribute1', 'attribute2', ...]  # Replace with actual attribute names
```

Then your code would look as follows when removing the duplicates in step 6 and in later steps when listing the specific column within the DataFrame called "data.”:

```python
# Removing Duplicates
data.drop_duplicates(inplace=True)

# Converting to Lowercase
data['headline'] = data['headline'].str.lower()

# Handling Missing Values
data['headline'].fillna("", inplace=True)
```

## **Step 5: Initial Data Exploration**

To understand the dataset better, let's start with some initial exploration. We'll look at the first few rows, check for missing values, and examine the data types.

```python
# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Examine data types
print(data.dtypes)
```

```python
0
0   1\tFed official says weak data caused by weather
1  2\tFed's Charles Plosser sees high bar for cha...
2  3\tUS open: Stocks fall after Fed official hin...
3            4\tFed risks falling 'behind the curve'
4  5\tFed's Plosser: Nasty Weather Has Curbed Job...
0    0
dtype: int64
0    object
dtype: object
```

## **Step 6: Preprocessing the Text Data**

Next, I will prepare and preprocess this dataset by removing any duplicate rows (essential when dealing with larger data sets), and then converting all the characters to lowercase to standardize the text and ensure that text comparison operations are case-insensitive. A valuable tool when cleaning data.

```python
# Removing Duplicates
data.drop_duplicates(inplace=True)

# Converting to Lowercase
data[0] = data[0].str.lower()

# Handling Missing Values
data[0].fillna("", inplace=True))
```

## **Step 7: Tokenize the News Article Headlines**

The next step, tokenizing the news headlines, is an essential step in processing

 text data for natural language processing (NLP) tasks. Tokenization breaks the text into individual words or "tokens," which makes it easier to analyze the text.

```python
# Tokenize the headlines
data['tokens'] = data[0].apply(word_tokenize)
```

## **Step 8: Text Vectorization**

To use the text data in a machine learning model, we need to convert the text into numerical vectors. Two common methods for this are Count Vectorization and TF-IDF Vectorization.

```python
# Initialize the CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the headlines
count_vectorized = count_vectorizer.fit_transform(data[0])

# Display the shape of the resulting matrix
print(count_vectorized.shape)
```

## **Step 9: Splitting the Data**

Now, we need to split the dataset into training and testing sets to train our model and evaluate its performance.

```python
# Split the data into features (X) and target (y)
X = count_vectorized
y = data['target_column']  # Replace 'target_column' with the actual column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## **Step 10: Building a Naive Bayes Classifier**

We'll build a simple Naive Bayes classifier to classify news articles based on their headlines. We'll use the Multinomial Naive Bayes algorithm.

```python
# Initialize the Naive Bayes classifier
naive_bayes = MultinomialNB()

# Train the classifier
naive_bayes.fit(X_train, y_train)

# Make predictions
y_pred = naive_bayes.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
```

This basic example should give you a good start on text cleaning and classification tasks using Python. As you gain more experience and explore more complex datasets, you can refine and expand on these techniques. Good luck with your text classification project!
