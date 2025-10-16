from sklearn.metrics import accuracy_score

# Actual labels (ground truth)
y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]

# Predictions from our model
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]


acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)
# Accuracy: 0.9
from sklearn.metrics import precision_score, recall_score

# Actual labels (ground truth)
y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]

# Predictions from our model
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
# Precision: 1.0
# Recall: 0.8333333333333334  
# Precision → Of all the "Yes" predictions, how many were correct?
# Recall → Of all the actual "Yes" cases, how many did we catch?

f1 = f1_score(y_true, y_pred)
print("F1-score:", f1)
# F1-score: 0.9090909090909091  This means the model is both precise (few false alarms) and fairly good at recall (found most positives).
# Harmonic mean of precision and recall 2*p*r/(p+r)
