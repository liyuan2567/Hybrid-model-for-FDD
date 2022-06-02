# Hybrid-model-for-FDD

## Motivation
* When dataset freshness is critical, the annotating of high speed unlabelled data streams becomes critical but remains an open problem.
* We propose PLStream, a novel Apache Flink-based framework for fast polarity labelling of massive data streams, like Twitter tweets or online product reviews.

## Environment Requirements
1. tensorflow 2.8
2. Python 3.7

## DataSource
* CWRU Dataset: https://course.fast.ai/datasets#nlp
* MFTP Dataset: https://course.fast.ai/datasets#nlp
### CWRU
* 1.6 million labeled Tweets:
* Source:[CWRU](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
### MFTP
* 280,000 training and 19,000 test samples in each polarity
* Source:[MFTP](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz)

## Quick Start
quick try PLStream on yelp review dataset
### Data Prepare
```
cd Hybrid-model-for-FDD
weget https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz
tar zxvf yelp_review_polarity_csv.tgz
mv yelp_review_polarity_csv/train.csv train.csv
```
### 1. Install required environment of PLStream
* please make sure Environment Requirements mentioned above is ready.
```
python CWRU.py
```
### 2. Run Hybrid-model approach on MFPT
```
python MFPT.py
```
