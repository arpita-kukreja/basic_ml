import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Input data
speed = [[20], [40], [60], [80], [100]]     # Features
mileage = [8, 15, 22, 20, 15]               # Labels

# Step 2: Transform into polynomial features
# TODO:
poly = PolynomialFeatures(degree=2)
speed_poly = poly.fit_transform(speed)

# Step 3: Create the model
model = LinearRegression()

# Step 4: Train the model
# TODO: complete this line
model.fit(speed_poly,mileage)

# Step 5: Make prediction for 120 km/h
predicted_mileage = model.predict(poly.transform([[120]]))
print(predicted_mileage)  # [5.2] output
