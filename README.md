# Profit Prediction Using Linear Regression

## Problem
Suppose you are the CEO of a Food Trucks franchise. You are considering different cities to open a new point of sale. The chain already has trucks in different cities and you have data for the city's profits and populations.
You want to use this data to help you choose the city to open a new point of sale.

[![dataset.png](https://i.postimg.cc/4yB2BTbQ/dataset.png)](https://postimg.cc/fSSvRpPV)

[![plot.png](https://i.postimg.cc/wBXVwp9R/plot.png)](https://postimg.cc/p9dzd3hP)

## Plan
* Get predictions using scipy library
* Get predictions using sklearn library
  
# Linear Regression Model

## Scipy
```python
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
```
[![scipy.png](https://i.postimg.cc/qvfyS5bs/scipy.png)](https://postimg.cc/nCkXQ0tr)

## Sklearn
```python
y_pred = regressor.predict(X_test)
```

Train test predictions<br>
[![skl1.png](https://i.postimg.cc/N0xs8pP7/skl1.png)](https://postimg.cc/JyGCMc8G)

Test test predictions<br>
[![skl2.png](https://i.postimg.cc/fLrh5F7X/skl2.png)](https://postimg.cc/5YLG2sRN)

## Compare Two Regressors

Plots<br>
[![compp.png](https://i.postimg.cc/m2MtzNRV/compp.png)](https://postimg.cc/RNSM2tWn)

R2_scores
```python
Scipy: 0.8378732325263409
SKLearn: 0.7221737943890659
```

Random Predictions
```python
Angers
Scipy: [14.357633877788937]
SkLearn: [14.12309067]
Delta = [0.23454321]
=============================
Paris
Scipy: [234.88990300623541]
SkLearn: [230.38999145]
Delta = [4.49991156]
=============================
Marseille
Scipy: [99.06302261525012]
SkLearn: [97.19017184]
Delta = [1.87285077]
```