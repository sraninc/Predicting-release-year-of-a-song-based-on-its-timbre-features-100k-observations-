# Predicting the release year of a song using its timbre features

## Introduction

### Background

The release year of songs can be predicted based on their time-specific “signatures” that are jointly determined by rhythm, dynamics, timbre, etc. Timbre is the character or quality of a musical sound or voice as distinct from its pitch and intensity. Songs released in different years can be differentiated based on their timbre values. Therefore, this project aims to predict a song's release year based on its timbre features.

### Dataset

We used a dataset of 100,000 audio samples from the year 1922 to the year 2010. The data is split into 90% training and 10% test data. The dataset consists of 90 predictor variables, of which the first 12 columns are timbre averages, and the remaining 78 columns are covariance variables. Response variable has two formats: one is in continuous form (from 1922 to 2010), and the other format is grouped in classes (“_prior to 1980_”, “_between 1980 - 2000_”, and “_after 2000_”). There is no missing data in the dataset.

### Purpose

This project aims to find a regression model that gives the best estimate for the release years based on 90 features and uses that model to predict the release year based on the timbre features of an audio sample. Additionally, this work aims to find a classification model that most accurately classifies the estimation based on 90 features and uses that model to predict the class of the release year based on the timbre features of an audio sample.

## Data Pre-Processing

### Standardization 

Centering and scaling is a data preprocessing/transformation technique that normalizes continuous variables to be in the same range. It is important to standardize the data before fitting the models that are influenced by distance metrics such as Support Vector Machines, Lasso, Ridge, etc. Data were not centered or scaled before building generative classification models. For this project, we standardized both train and test data for analysis.

### Outliers

We used boxplots to review the outliers. While we did not take any steps to remove specific outliers from the data, we acknowledge that this may contribute to poorer performance in some models. Tree-based methods may be our best-performing models, which are more robust to outliers.

### Dimension reduction
In this technique, model variance is controlled by transforming the original predictors to obtain new ones and using them as covariates in the regression model. 
 - First, we utilized the dimensionality-reduction method Principal Components Analysis (PCA) to reduce the dimensionality of the large data set. However, the first two components explained only 19% of the variance, and the first 55 principal components explained 90% of the variance.
 - Second, we utilized the Principal Component Regression (PCR) model as it overcomes the limitations of PCA by taking into account the class variable and mitigating the overfitting of the data. However, we obtained similar results where the first 50 components explained 87% of the variance.
 - Finally, we utilized the supervised approach of Partial Least Squares (PLS), which can help reduce bias but has the potential to increase variance. The results from this method did not significantly reduce the dimensions, where 29 components explained only 62% of the variance.

### Variable selection & Shrinkage
Variable selection is selecting relevant/significant variables corresponding to a response to reduce the model complexity and the variability in the estimates. Shrinkage methods such as lasso and ridge reduce model variance by shrinking regression coefficients toward zero. The forward stepwise selection approach considers small sets of models and then builds up to the whole dataset, which presents a suitable approach to deal with a large dataset. With this approach, we had the best 77 predictors for the continuous response. These predictors were used in further training of the models. Lasso regression resulted in 89 best predictors using min lambda and 57 best predictors using min lambda within one SE from min lambda. For the categorical response, forward subset selection resulted in a subset of 53 predictors, and lasso-based shrinkage resulted in a subset of 56 predictors.

### Downsampling
Class _prior to 1980_ has significantly fewer observations which could affect model performance for this class. We applied a downsampling technique and downsampled each class to the number of samples of the smallest class. Class distribution after downsampling results in 7815 observations in each class.

## Machine Learning Algorithms:

1) **Regression algorithms**: Lasso | Support Vector Regression | Random Forest | Extreme Gradient Boosting | Stacking | ANN
2) **Classification algorithms**: Linear Discriminant Analysis(LDA) | Quadratic Discriminant Analysis(QDA) | Naive Bayes | K-Nearest Neighbors(KNN) | Support Vector Classifiers (SVC) | Logistics Regression | Stacking

## Model Evaluation: 
### Regression algorithms
**Metric:** Mean Squared Error(MSE)
**Best-performed Regression Model:** Based on the simplicity and interpretability of the model, lasso regression performed the best even though the test error was not the best value it is comparable to the best MSE achieved. However, based on our key performance metric MSE value and robustness to outliers, XG Boost performed the best by achieving the lowest MSE of 87.05.

### Classification algorithms
**Metric:** Sensitivity and Specificity
**Best-performed Classification Model:** XGBoost classification algorithm performed the best among other algorithms for given downsampled data with obtained Accuracy = 0.7. XGBoost is robust to outliers and is ideal for data with many observations, so it is not surprising it performed well here, accurately classifying 70% of the test observations. Specificity and sensitivity results were also high for XGboost compared to others.




