##q.1

import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Car_Model": ["Swift", "Baleno", "Creta", "Thar", "Fortuner"],
    "Engine_Size": [1.2, 1.2, 1.5, 2.0, 2.8],
    "Fuel_Efficiency": [22, 23, 17, 15, 10]
}

df = pd.DataFrame(data)

print("Car DataFrame:")
print(df)

colors = ["red", "blue", "green", "orange", "purple"]

plt.scatter(df["Engine_Size"], df["Fuel_Efficiency"],
            c=colors, s=100)

for i in range(len(df)):
    plt.annotate(
        df["Car_Model"][i],
        (df["Engine_Size"][i], df["Fuel_Efficiency"][i])
    )

plt.title("Engine Size vs Fuel Efficiency")
plt.xlabel("Engine Size (Litres)")
plt.ylabel("Fuel Efficiency (km/L)")
plt.grid(True)
plt.show()

##q.2

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = {
    "Training_Hours_Per_Week": [2, 4, 5, 7, 8, 10, 12, 14, 15, 18],
    "Performance_Score": [30, 40, 45, 55, 60, 65, 75, 80, 85, 95]
}

df = pd.DataFrame(data)

print("Sports Performance Dataset:")
print(df)

X = df[["Training_Hours_Per_Week"]]
y = df["Performance_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Performance Scores:")
print(y_test.values)

print("\nPredicted Performance Scores:")
print(y_pred.round(2))

print("\nR2 Score:", r2_score(y_test, y_pred))

plt.scatter(X, y, color="blue", label="Actual Performance")

plt.plot(X, model.predict(X), color="red",
         label="Regression Line")

plt.xlabel("Training Hours Per Week")
plt.ylabel("Performance Score")
plt.title("Training Hours vs Sports Performance")
plt.legend()
plt.grid(True)
plt.show()