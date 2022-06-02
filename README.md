# Hybrid-model-for-FDD

## Motivation
* Induction motors are widely used in industrial applications. However, unpredictable failures of the motors may cause high financial losses to the manufacturers and expensive repair time and energy, especially in the scenarios that require high real-time response. These failures of rolling bearings can lead to reduced performance of these machines and even accidents, resulting in extremely expensive economic losses.
* The main objective of our work is to develop a novel FDD system with hybrid-model approach to improve the efficiency while sustaining high accuracy.

## Environment Requirements
1. tensorflow 2.8
2. Python 3.7

## Hybrid-model approach
#### Machine Learning + Neural Network 
* SVM/RF/KNN + CNN
#### Machine Learning + Machine Learning
* SVM/RF/KNN + SVM/RF/KNN

## Overview
![avatar](/overview.png)

## Data Description
### CWRU
| Type     | Load 0  | Load 1  | Load 2 | Load 3  |
| -------- | ------- | ------- | ------ | ------- |
| Training | 1955    | 2181    | 2184   | 2187    |
| Testing  | 501     | 559     | 560    | 561     |
* Source:[CWRU](http://csegroups.case.edu/bearingdatacenter/home))
### MFTP
| Type     | Number of data |
| -------- | -------------- |
| Training | 4296           |
| Testing  | 1101           |
* Source:[MFTP](https://mfpt.org/fault-data-sets/))

## Quick Start
quick try Hybrid-model approach
### 1. Run Hybrid-model approach on CWRU
* please make sure Environment Requirements mentioned above is ready.
```
python CWRU.py
```
### 2. Run Hybrid-model approach on MFPT
```
python MFPT.py
```
