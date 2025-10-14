import numpy as np

# Step 1: Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Step 2: Example raw scores (like the linear step inside Logistic Regression)
scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

# Step 3: Convert scores to probabilities using sigmoid
probabilities = sigmoid(scores)

# Step 4: Apply threshold (0.5) to get final predictions
predictions = (probabilities >= 0.5).astype(int)

# Display results
for s, p, pred in zip(scores, probabilities, predictions):
    print(f"Score: {s:>4} -> Probability: {p:.2f} -> Prediction: {pred}")
  '''
  Output
Score: -2.0 -> Probability: 0.12 -> Prediction: 0
Score: -1.0 -> Probability: 0.27 -> Prediction: 0
Score:  0.0 -> Probability: 0.50 -> Prediction: 1
Score:  1.0 -> Probability: 0.73 -> Prediction: 1
Score:  2.0 -> Probability: 0.88 -> Prediction: 1
  '''
