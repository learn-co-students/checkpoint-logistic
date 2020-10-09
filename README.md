# Logistic Regression


```python
# Run this cell without changes
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
```

### 1) Why is logistic regression typically better than linear regression for modeling a binary target/outcome?

=== BEGIN MARK SCHEME ===

Any one or more of the following:

Logistic regression will never predict probabilities greater than 1 or less than 0, unlike linear regression.

Logistic regression often provides a closer fit to the conditional means of 0/1 target values than linear regression does.

Logistic regression provides for interpretations of coefficients in terms of log odds, which can be relevant for some problems.

=== END MARK SCHEME ===

![cnf matrix](visuals/cnf_matrix.png)

### 2) Using the confusion matrix above, calculate precision, recall, and F-1 score.
Show your work, not just your final numeric answer.  For example, write `answer = 10 / 20` if 10 and 20 are the relevant values, instead of writing `answer = 0.5`.

Use the variables provided (`precision`, `recall`, `f1_score`)


```python
# Replace None with appropriate code
precision = None
### BEGIN SOLUTION

from test_scripts.test_class import Test
test = Test()

precision = 30/(30+4)

test.save(precision, "precision")

### END SOLUTION
print('Precision: {}'.format(precision))
```

    Precision: 0.8823529411764706



```python
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS

# precision should be a floating point number
assert type(precision) == float or type(precision) == np.float64

### BEGIN HIDDEN TESTS

from test_scripts.test_class import Test
test = Test()

precision_test = test.load_ind("precision")
assert np.isclose(precision, precision_test)

### END HIDDEN TESTS
```


```python
# Replace None with appropriate code
recall = None
### BEGIN SOLUTION

from test_scripts.test_class import Test
test = Test()

recall = 30 / (30 + 12)

test.save(recall, "recall")

### END SOLUTION
print('Recall: {}'.format(recall))
```

    Recall: 0.7142857142857143



```python
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS

# recall should be a floating point number
assert type(recall) == float or type(recall) == np.float64

### BEGIN HIDDEN TESTS

from test_scripts.test_class import Test
test = Test()

recall_test = test.load_ind("recall")
assert np.isclose(recall, recall_test)

### END HIDDEN TESTS
```


```python
# Replace None with appropriate code
f1_score = None
### BEGIN SOLUTION

from test_scripts.test_class import Test
test = Test()

f1_score = 2 * (precision * recall) / (precision + recall)

test.save(f1_score, "f1_score")

### END SOLUTION
print('F-1 Score: {}'.format(f1_score))
```

    F-1 Score: 0.7894736842105262



```python
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS

# f1_score should be a floating point number
assert type(f1_score) == float or type(f1_score) == np.float64

### BEGIN HIDDEN TESTS

from test_scripts.test_class import Test
test = Test()

f1_test = test.load_ind("f1_score")
assert np.isclose(f1_score, f1_test)

### END HIDDEN TESTS
```

### 3)  What is a real life example of when you would care more about recall than precision? Make sure to include information about errors in your explanation.

=== BEGIN MARK SCHEME ===

We would care more about recall than precision in cases where a Type II error (a False Negative) would have serious consequences. An example of this would be a medical test that determines if someone has a serious disease.

A higher recall would mean that we would have a higher chance of identifying all people who ACTUALLY had the serious disease.

=== END MARK SCHEME ===

<img src = "visuals/many_roc.png" width = "700">

### 4) Which model's ROC curve from the above graph is the best? Explain your reasoning.

Note: each ROC curve represents one model, each labeled with the feature(s) inside each model.

The three models are:

1. **Age** (green curve)
2. **Estimated Salary** (orange curve)
3. **All Features** (pink curve)

=== BEGIN MARK SCHEME ===

The best ROC curve in this graph is for the one that contains all features (the pink one). This is because it has the largest area under the curve. 

The ROC curve is created by obtaining the ratio of the True Positive Rate to the False Positive Rate over all thresholds of a classification model.

=== END MARK SCHEME ===

### Logistic Regression Example

The following cell includes code to train and evaluate a model


```python
# Run this cell without changes

# load data
network_df = pickle.load(open('write_data/sample_network_data.pkl', 'rb'))

# partion features and target 
X = network_df.drop('Purchased', axis=1)
y = network_df['Purchased']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# scale features
scale = StandardScaler()
scale.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)

# build classifier
model = LogisticRegression(C=1e5, solver='lbfgs')
model.fit(X_train, y_train)

# get the accuracy score
print(f'The classifier has an accuracy score of about {round(model.score(X_test, y_test), 3)}.')
```

    The classifier has an accuracy score of about 0.971.


### 5) The model above has an accuracy score that might be too good to believe. Using `y.value_counts()`, explain how `y` is affecting the accuracy score.


```python
# Run this cell without changes
y.value_counts()
```




    0    257
    1     13
    Name: Purchased, dtype: int64



=== BEGIN MARK SCHEME ===

`y.value_counts()` indicates that we have a class imbalance. When we have class imbalance our model will predict the most common class preferentially and is not penalized for doing so in the accuracy score, because it is still getting the right answer most of the time. 

=== END MARK SCHEME ===

### 6) What is one method you could use to improve your model to address the issue discovered in Question 5?

=== BEGIN MARK SCHEME ===

Any one of these is an acceptable answer:

 - Use SMOTE to generate additional synthetic data points for the minority class to create class balance.
 - Oversample (with replacement) from the minority class to create class balance.
 - Use class weights to place more emphasis on the minority class when fitting models.
 - Use precision or recall metrics to inform model decisions/hyperparameter tuning.

NOTE: Undersampling is not a valid answer because there are so few positive cases

=== END MARK SCHEME ===
