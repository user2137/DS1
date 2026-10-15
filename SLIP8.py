##q.1

import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Latitude": [35.68, 19.07, 38.32, -6.20, 27.72, 12.97, -3.74, 36.20],
    "Longitude": [139.69, 72.87, 142.36, 106.85, 85.32, 77.59, -73.25, 59.60],
    "Depth": [10, 15, 25, 30, 12, 18, 40, 20],
    "Magnitude": [4.5, 5.2, 6.8, 4.1, 5.8, 3.9, 7.2, 6.1]
}

df = pd.DataFrame(data)

print("Earthquake Dataset:")
print(df)

plt.figure(figsize=(12, 6))

scatter = plt.scatter(
    df["Longitude"],
    df["Latitude"],
    c=df["Magnitude"],
    cmap="hot",
    s=df["Magnitude"] * 50,
    alpha=0.8,
    edgecolors="black"
)

plt.colorbar(scatter, label="Earthquake Magnitude")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geospatial Visualization of Earthquakes")

plt.grid(True)
plt.show()

##q.2

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    "YearsExperience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [30000, 35000, 40000, 45000, 50000,
               60000, 65000, 70000, 80000, 85000]
}

df = pd.DataFrame(data)

print("Salary Dataset:")
print(df)

X = df[["YearsExperience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Salary:")
print(y_test.values)

print("\nPredicted Salary:")
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

plt.scatter(X, y, color="blue", label="Actual Salary")
plt.plot(
    X,
    model.predict(X),
    color="red",
    label="Regression Line"
)

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Linear Regression")
plt.legend()
plt.show()