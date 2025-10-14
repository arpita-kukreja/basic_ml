import pandas as pd
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5],
    "Pass": [0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[["Study_Hours"]]   # Feature
y = df["Pass"]            # Label

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Predictions on the same dataset that we took
predictions = model.predict(X)
print("Predictions:", predictions)

# Probabilities
probabilities = model.predict_proba(X)
print("Probabilities:\n", probabilities)
'''
output-
Predictions: [0 0 1 1 1]
Probabilities:
 [[0.8157613  0.1842387 ]
 [0.60836998 0.39163002]
 [0.35275336 0.64724664]
 [0.16051755 0.83948245]
 [0.06286686 0.93713314]]
'''
