# Twitter Sentiment Analysis

## Overview

Social media has received more attention nowadays. Public and private opinion about a wide variety of subjects are expressed and spread continually via numerous social media. Twitter has been growing in popularity and is used every day by people to express opinions about different topics, such as products, movies, music, politicians, events, social events, among others.

This project addresses the problem of sentiment analysis on Twitter; that is, classifying tweets according to the sentiment expressed in them: positive, negative, or neutral. Due to the large amount of usage, we hope to achieve a reflection of public sentiment by analyzing the sentiments expressed in the tweets.

Analyzing public sentiment is important for many applications such as firms trying to find out the response of their products in the market, predicting political elections, and predicting socioeconomic phenomena like stock exchange. The aim of this project is to develop a functional classifier for accurate and automatic sentiment classification of an unknown tweet stream. We apply deep learning techniques to classify sentiment of Twitter data. The two deep learning techniques used are Long Short Term Memory (LSTM) and Dynamic Convolutional Neural Network (DCNN).

## Dataset

The dataset used for this project can be found on Kaggle: [Twitter Sentiment Analysis](https://www.kaggle.com/ywang311/twitter-sentiment)

## Project Structure

- `SentimentAnalysis.ipynb`: Jupyter Notebook containing the code for data processing, model building, training, and evaluation.
- `README.md`: This file, providing an overview of the project and instructions for running the code.

## Requirements

- Python 3.x
- Jupyter Notebook
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras
