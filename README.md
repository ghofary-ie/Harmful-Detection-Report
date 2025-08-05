# Harmful Content Detection Pipeline

Project by: Abdul Ghofar for personal portfolio.


# Project Description

In today’s digital age, content moderation plays a crucial role in protecting users and maintaining the integrity of online platforms. The increasing volume of user generated content particularly on social media presents both opportunities and challenges. Among the most pressing challenges is the spread of harmful content such as hate speech, misinformation, graphic violence, and abusive language.

Traditional moderation systems often struggle with scale, context, and cultural nuance. To address this, our project introduces a hybrid moderation workflow that combines Machine Learning (ML) with human review to improve both accuracy and efficiency. This system is designed to detect harmful content, identify emerging threat indicators such as evolving slurs, code words, or trending hashtags, and support proactive content moderation efforts.

The ultimate goal is to help Trust & Safety teams stay ahead of online harms by reducing false positives and negatives, adapting to new threats in real time, and ensuring scalable moderation processes. This project emphasizes ethical AI use, contextual understanding, and operational feasibility key components for creating safer online spaces.


# Project Objective

The primary goal is to develop an effective and adaptive moderation pipeline that combines the strengths of machine learning and human review. By integrating automated classifiers with manual assessment, the system aims to improve accuracy in detecting harmful content across social media platforms.

Beyond detection, the project also seeks to proactively identify linguistic patterns, trending hashtags, and OSINT (Open-Source Intelligence) signals that may indicate emerging threats. This dual approach supports both real-time moderation and strategic foresight for online safety teams.


# Workflow

This project demonstrates a complete workflow for detecting harmful content on social media using machine learning models combined with human review.

The dataset used originates from Kaggle.com and is intended for research and educational purposes.

- Loads and preprocesses raw comment data
- Applies transformer-based toxicity models (e.g., Detoxify, HuggingFace)
- Integrates human review labels to validate and improve model predictions
- Identifies false positives and false negatives for error analysis
- Provides clear evaluation metrics and visualizations
- Highlights real-world challenges in content moderation

**Note on Data Usage:**
All efforts are made to handle the data responsibly, respecting user privacy and platform policies. Sensitive content is treated with care, and this project is intended solely for educational and portfolio purposes.


# Data Desciption

The dataset includes about 8,000 tweets collected over one week (January 1–7, 2025). Each tweet is an anonymized user comment with details like the time it was posted. The tweets include all kinds of content some normal, some potentially harmful, such as threats, insults, or rude language. The data is cleaned and user privacy is protected. This dataset helps us study harmful content and build tools to detect it. We handle all data carefully and responsibly.


# Data Processing

Preparing for Data Processing using Python: 
1. Instaling the libraries

Run this first to make sure all the needed packages are available. Especially important in Colab or a new setup.

```python
!pip install detoxify tqdm transformers datasets torch
```

Below are the libraries used in this project and their purposes:

```python

import pandas as pd                                                      # Data manipulation and analysis with DataFrames
import numpy as np                                                       # Numerical operations and array handling
import matplotlib.pyplot as plt                                          # Plotting static and interactive visualizations
import seaborn as sns                                                    # Statistical data visualization built on matplotlib
from sklearn.metrics import classification_report, confusion_matrix      # Model evaluation metrics
from detoxify import Detoxify                                            # Pre-trained model for toxic content detection
from transformers import pipeline                                        # Access to transformer based NLP models (HuggingFace)
from tqdm import tqdm                                                    # Progress bars for loops and iterations
```

2. Importing the Dataset

This snippet mounts Google Drive in Colab, then loads the CSV dataset containing Twitter data for the first week of January 2025 into a pandas DataFrame for processing and analysis.

```python

from google.colab import drive               # To mount Google Drive in Colab
import pandas as pd                          

file_path = '/content/drive/MyDrive/Notebook Twitter Project/Raw-Data-Week-1,Jan-2025.csv'

df = pd.read_csv(file_path)

df.head()
```

3. Data Cleaning
   
This code removes duplicate tweets based on the tweet text to ensure each tweet is unique. After cleaning, it prints the total number of remaining tweets for transparency and quality assurance.

```python
df = df.drop_duplicates(subset='text')

print(f"Dataset size after cleaning: {len(df)} tweets")
```

4. Load and Apply Toxicity Models for Prediction

In this step, I apply two machine learning models to detect harmful content in tweets:

- Detoxify Model

Install and load the Detoxify model, which predicts multiple toxicity related scores for each tweet.

The model assigns scores for toxicity, severe toxicity, obscenity, threat, insult, and identity attack.

I loop through each tweet, score it, and add the results back into our main DataFrame.

Tweets with toxicity scores above a chosen threshold (0.8) are flagged as potentially harmful.

- HuggingFace Toxic BERT Model

Install and load the Toxic BERT model from HuggingFace, a powerful transformer based text classification model.

This model cross checks the tweets by classifying each as "toxic" or "non-toxic" with a confidence score.

I process each tweet (limiting input length to 512 tokens), storing the label and score in the DataFrame for comparison with Detoxify results.

```python
from detoxify import Detoxify
from transformers import pipeline
from tqdm import tqdm
import pandas as pd

detoxify_model = Detoxify('original')                                        # Load Detoxify model

scores = []                                                                  # Score tweets using Detoxify
for tweet in tqdm(df['tweet'], desc="Scoring tweets"):
    score = detoxify_model.predict(tweet)
    scores.append(score)

scores_df = pd.DataFrame(scores)                                             # Combine Detoxify scores with the original DataFrame
df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)

toxicity_threshold = 0.8                                                     # Flag tweets with toxicity > 0.8 as harmful
df['ml_flagged'] = df['toxicity'] > toxicity_threshold
print(df['ml_flagged'].value_counts())

classifier = pipeline("text-classification", model="unitary/toxic-bert")     # Load HuggingFace Toxic-BERT model

toxic_labels = []                                                            # Cross-check tweets with Toxic-BERT
toxic_scores = []
for tweet in tqdm(df['tweet'], desc="Cross-checking with Toxic-BERT"):
    try:
        result = classifier(tweet[:512])[0]
        toxic_labels.append(result['label'])
        toxic_scores.append(result['score'])
    except Exception:
        toxic_labels.append("error")
        toxic_scores.append(0)

df['toxicbert_label'] = toxic_labels                                         # Add Toxic-BERT results to DataFrame
df['toxicbert_score'] = toxic_scores

df.head()                                                                    # Preview updated DataFrame
```

5. Exporting Results for Human Review

After processing and scoring the tweets with ML models, the results need to be reviewed manually to validate and improve accuracy. This step saves the current dataset—including model predictions and metadata—as a timestamped CSV file in your Google Drive. This makes it easy to share and track different versions of the data for human annotation and further analysis.

```python
import datetime                                                               # Saves the processed DataFrame as a CSV file with a timestamp in the filename (optional)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f'/content/drive/MyDrive/Notebook Twitter Project/HumanReview{timestamp}.csv'

df.to_csv(output_path, index=False)
print(f"Saved processed data to {output_path}")
```

6. Human Review and Manual Validation

In this step, I focus on validating the model’s harmful content predictions by performing a manual review. I filter tweets that the model flagged as harmful based on Detoxify and HuggingFace models judgement, then review each tweet to confirm whether it truly violates community guidelines (X Rules and Policies). The manual judgment categorizes tweets as either "harmful" or "safe" in a Final Review column.

The criteria for this review are aligned with specific sections of the community rules and policies, focusing on severe toxicity, obscenity, threat, insult, and identity attack. This process helps improve the overall accuracy by identifying false positives and false negatives, which the model alone might miss or misclassify.

After reviewing the tweets flagged as harmful by the model, the next step involves manually checking tweets that the model marked as safe. This is done by searching for critical keywords (e.g., "Nigga", "Porn", "Child", "Teenager", "Sex") that frequently appear in harmful tweets identified earlier.

Using this targeted keyword search allows us to catch harmful content that the model may have missed. Based on this review, each tweet is labeled as "harmful" or "safe" in Final Review column. This additional step ensures a more comprehensive detection process by combining automated filtering with human insight.


5. Comparing Model Predictions with Human Review

This section compares the harmful content labels predicted by the machine learning models with the labels assigned through manual human review. It helps to evaluate the model's accuracy by identifying agreements, false positives (model flagged safe content as harmful), and false negatives (model missed harmful content). The comparison supports improving detection performance and understanding the model’s strengths and weaknesses.

```python
import pandas as pd
raw_path = '/content/drive/MyDrive/Notebook Twitter Project/RawData.csv'             # Load the raw model predictions and human review datasets from Google Drive
review_path = '/content/drive/MyDrive/Notebook Twitter Project/HumanReview20250721_230147 - Clean Data.csv'

df_raw = pd.read_csv(raw_path)
df_review = pd.read_csv(review_path)
```
Ensures column names are uniform, avoiding errors during merge due to mismatched column names.
```python
df_raw.columns = df_raw.columns.str.strip().str.lower()                                # Clean column names by removing whitespace and converting to lowercase for consistent merging
df_review.columns = df_review.columns.str.strip().str.lower()
```
Merges the two datasets using the unique identifier id, keeping only the common rows (how='inner').
```python 
df = pd.merge(df_raw, df_review[['id', 'human review']], on='id', how='inner')
```
Cleans and standardizes the label text to avoid mismatch during comparison and maps the model’s harmful/not harmful labels to human review’s violation/safe.
```python
df['transformer_harmful'] = df['transformer_harmful'].str.lower().str.strip()
df['human review'] = df['human review'].str.lower().str.strip()

df['transformer_harmful'] = df['transformer_harmful'].replace({
    'harmful': 'violation',
    'not harmful': 'safe'
})
```
This part creates three new columns in the dataframe to help analyze the agreement between the model's predictions and the human review labels, These columns are key to quantitatively assessing the model's errors and accuracy by breaking down where it agrees or disagrees with human judgment.

```python
df['match'] = df['transformer_harmful'] == df['human review']                           # Create boolean columns to identify agreement and disagreement between model and human labels
df['false_positive'] = (df['human review'] == 'safe') & (df['transformer_harmful'] == 'violation')
df['false_negative'] = (df['human review'] == 'violation') & (df['transformer_harmful'] == 'safe')
```
Displays the total number of evaluated samples and counts of correct predictions and errors.
```python 
print(f"Total Samples: {len(df)}")                                                      # Print evaluation summary statistics
print("Evaluation Summary:")
print(f"Correct Matches: {df['match'].sum()}")
print(f"False Positives (Model says Violation, human says Safe): {df['false_positive'].sum()}")
print(f"False Negatives (Model says Safe, human says Violation): {df['false_negative'].sum()}")
```
Saves the dataframe including evaluation results and combined dataset with evaluation columns back to Google Drive for record-keeping and further analysis
```python
output_path = '/content/drive/MyDrive/Notebook Twitter Project/Final_evaluation_result.csv'
df.to_csv(output_path, index=False)
print(f"\n Saved to: {output_path}")
```
Filters and displays tweets where the model made incorrect predictions for deeper manual inspection.
```python
false_positives = df[df['false_positive']]
display(false_positives[['tweet', 'transformer_harmful', 'human review']])

false_negatives = df[df['false_negative']]
display(false_negatives[['tweet', 'transformer_harmful', 'human review']])
```
Filters and displays tweets where the model made incorrect predictions for deeper manual inspection and to Show example false positives and false negatives with relevant columns to inspect the errors manually
```python
false_positives = df[df['false_positive']]
display(false_positives[['tweet', 'transformer_harmful', 'human review']])

false_negatives = df[df['false_negative']]
display(false_negatives[['tweet', 'transformer_harmful', 'human review']])
```


