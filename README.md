# Profit Prediction Using Regression & Machine Learning

- [Profit Prediction Using Regression & Machine Learning](#profit-prediction-using-regression--machine-learning)
  - [Problem](#problem)
- [Linear Regression Model](#linear-regression-model)
  - [Scipy](#scipy)
  - [Sklearn](#sklearn)
  - [Compare Two Regressors](#compare-two-regressors)
- [Support Vector Machine Model](#support-vector-machine-model)
  - [Plots](#plots)
  - [R2 Score](#r2-score)
  - [Random Predictions](#random-predictions)
- [Polynomial Regression Model](#polynomial-regression-model)
  - [Plots](#plots-1)
  - [R2 Score](#r2-score-1)
  - [Random Predictions](#random-predictions-1)
- [Decision Trees Model](#decision-trees-model)
  - [Plots](#plots-2)
  - [R2 Score](#r2-score-2)
- [Random Forests Model](#random-forests-model)
  - [Plots](#plots-3)
  - [R2 Score](#r2-score-3)
- [Web Scraping](#web-scraping)
- [Graphical User Interface (GUI - Tkinter)](#graphical-user-interface-gui---tkinter)
- [Conclusion](#conclusion)

## Problem
Suppose you are the CEO of a Food Trucks franchise. You are considering different cities to open a new point of sale. The chain already has trucks in different cities and you have data for the city's profits and populations.
You want to use this data to help you choose the city to open a new point of sale.

[![dataset.png](https://i.postimg.cc/4yB2BTbQ/dataset.png)](https://postimg.cc/fSSvRpPV)

[![plot.png](https://i.postimg.cc/wBXVwp9R/plot.png)](https://postimg.cc/p9dzd3hP)
  
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
print("Scipy: "+str(r_value))
print("SKLearn: "+str(r2_score(y_train, regressor.predict(X_train))))
==========================================================================
Scipy: r2 = 0.8378732325263409
SKLearn: r2 = 0.7221737943890659
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

## Plots
[![svm.png](https://i.postimg.cc/wMSyyyJn/svm.png)](https://postimg.cc/7JnYjLKN)

## R2 Score
```python
print(r2_score(sc_y.inverse_transform(y), sc_y.inverse_transform(regressor.predict(sc_X.transform(sc_X.inverse_transform(X))))))
==============================================================================
r2 = 0.6931820237472561
```

## Random Predictions
```python
150K (Angers)
Predict: [12.70672805]
=============================
215K (Rennes)
Predict: [20.49885932]
```
# Polynomial Regression Model
```python
from sklearn.preprocessing import PolynomialFeatures
poly_reg4 = PolynomialFeatures(degree=4)
X_poly4 = poly_reg4.fit_transform(X)
lin_reg4 = LinearRegression()
lin_reg4.fit(X_poly4, y)
```

## Plots
[![POOOOOOOOOOOOOOOO.png](https://i.postimg.cc/KvwwjZWL/POOOOOOOOOOOOOOOO.png)](https://postimg.cc/Vd9DHcmL)

## R2 Score
```python
print(r2_score(y, lin_reg.predict(X)))
==========================================
r2 = 0.7020315537841397
```

## Random Predictions
```python
150K (Angers)
Predict lin: [[12.56809849]]
Predict poly: [[1.0000e+00 1.5000e+01 2.2500e+02 3.3750e+03 5.0625e+04]]
=============================
215K (Rennes)
Predict lin: [[17.59533788]]
Predict poly: [[1.00000e+00 2.10000e+01 4.41000e+02 9.26100e+03 1.94481e+05]]
```

# Decision Trees Model
```python
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
```

## Plots
[![DECISIONTOOOOOOP.png](https://i.postimg.cc/053Wcswc/DECISIONTOOOOOOP.png)](https://postimg.cc/mhYygKVH)

## R2 Score
```python
print(r2_score(y, regressor.predict(X)))
===========================================
r2 = 1
```

# Random Forests Model
```python
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)
```

## Plots
[![RANDOMTOOOOOOOP.png](https://i.postimg.cc/kgjfRPqQ/RANDOMTOOOOOOOP.png)](https://postimg.cc/gxZv15wj)
## R2 Score
```python
print(r2_score(y, regressor.predict(X)))
===========================================
r2 = 0.9302612028627444
```

# Web Scraping
I scraped this [Wikipedia page](https://fr.wikipedia.org/wiki/Liste_des_communes_de_France_les_plus_peupl%C3%A9es) to get all cities with more than 30k people.
```python
response = requests.get("https://fr.wikipedia.org/wiki/Liste_des_communes_de_France_les_plus_peupl%C3%A9es")
content = response.content
```
```python
cities = []
for i in range(2, 277):
  cities.append(trs[i].text.split("\n")[3])
population = []
for i in range(2, 277):
  if(i==173):
    population.append(trs[i].text.split("\n")[9])
  else:
    population.append(trs[i].text.split("\n")[8])
```
```python
zip_iterator = zip(cities, population)
final_dict = dict(zip_iterator)
```
Now we just have to ask a city to the user, the algorithm will check if the city is included on the dictionnary keys. If it is the case, I get the correponding population to predict the profit by the Decision Tree Regressor.<br>
```python
prediction = DTR.predict([[convIntPop]])
prediction = float(str(prediction)[1:-1])
prediction*=10000
print("Prediction Decision Tree: "+str(desiredCity)+" -> "+str(prediction)+"$")
```

# Graphical User Interface (GUI - Tkinter)
To improve the user experience, I created a simple GUI with Tkinter.
The CEO can easily enter a City and note the predicted profit given by the machine learning model.<br>
```python
gui = Tk(className='Food Truck Project')
gui.geometry("200x200")
label = Label(gui, text="Profit Predictor")
cityEntry = Entry(gui, width=20)
validBtn = Button(gui, text="Valid", command=predictResult)
```
As you can see with this code, I made a very simple GUI to display predictions.<br>
The final result is printed in a label:<br>
```python
textLabel=f"Prediction: \n{city} -> {prediction}$"
```
[![food.png](https://i.postimg.cc/tCfzfpJS/food.png)](https://postimg.cc/BXTDPWxH)


# Conclusion
|    | Linear | Polynomial | SVM | Decision Tree | Random Forest |
|-------|---------|------------|-----|---------------|---------------|
|r2score|0.84|0.70|0.69|1|0.93|
|model|[![skl1.png](https://i.postimg.cc/N0xs8pP7/skl1.png)](https://postimg.cc/JyGCMc8G)|[![POOOOOOOOOOOOOOOO.png](https://i.postimg.cc/KvwwjZWL/POOOOOOOOOOOOOOOO.png)](https://postimg.cc/Vd9DHcmL)|[![svm.png](https://i.postimg.cc/wMSyyyJn/svm.png)](https://postimg.cc/7JnYjLKN)|[![DECISIONTOOOOOOP.png](https://i.postimg.cc/053Wcswc/DECISIONTOOOOOOP.png)](https://postimg.cc/mhYygKVH)|[![RANDOMTOOOOOOOP.png](https://i.postimg.cc/kgjfRPqQ/RANDOMTOOOOOOOP.png)](https://postimg.cc/gxZv15wj)|


The most acurate model is the <b>Decision Tree</b> one, it has <b>100%</b> of precision (strange ?).<br>

[![DECISIONTOOOOOOP.png](https://i.postimg.cc/053Wcswc/DECISIONTOOOOOOP.png)](https://postimg.cc/mhYygKVH)