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
150K (Angers)
Scipy: [13.999723784532057]
SkLearn: [13.772103]
Delta = [0.22762078]
=============================
215K (Rennes)
Scipy: [21.15792564966962]
SkLearn: [20.79185634]
Delta = [0.36606931]
```

# Support Vector Machine Model

```python
from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(X, y)
```
## Observe predictions

Plots<br>
[![svm.png](https://i.postimg.cc/fL8dw11J/svm.png)](https://postimg.cc/WFJz8YNv)

Random Predictions
```python
150K (Angers)
Predict: [12.70672805]
=============================
215K (Rennes)
Predict: [20.49885932]
```
