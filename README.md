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

