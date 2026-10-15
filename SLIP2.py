##q.1

import pandas as pd

data = {
    "Person_ID": [1, 2, 3, 4, 5, 6],
    "Age": [18, 22, 27, 35, 45, 60]
}

age_records = pd.DataFrame(data)

print("Original Data:")
print(age_records)


age_records["Age_Group"] = pd.cut(
    age_records["Age"],
    bins=3,
    labels=["Young", "Middle", "Old"]
)

print("\nDiscretized Data:")
print(age_records)

##q.2

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = {
    "likes": [
        100, 200, 300, 400, 500,
        600, 700, 800, 900, 1000,
        1100, 1200, 1300, 1400, 1500,
        1600, 1700, 1800, 1900, 2000
    ],
    "views": [
        1000, 1800, 2500, 3500, 4200,
        5000, 5800, 6500, 7500, 8200,
        9000, 9800, 10500, 11500, 12200,
        13000, 13800, 14500, 15500, 16200
    ]
}

df = pd.DataFrame(data)

print("YouTube Dataset:")
print(df)


X = df[["likes"]]
y = df["views"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nActual Views:")
print(y_test.values)

print("\nPredicted Views:")
print(y_pred.round(2))

print("\nMean Squared Error:",
      mean_squared_error(y_test, y_pred))

print("R2 Score:",
      r2_score(y_test, y_pred))


plt.scatter(X_test["likes"], y_test,
            color="blue", label="Actual Views")

sorted_data = X_test.copy()
sorted_data["Predicted_Views"] = y_pred
sorted_data = sorted_data.sort_values("likes")

plt.plot(sorted_data["likes"],
         sorted_data["Predicted_Views"],
         color="red", label="Regression Line")

plt.xlabel("Likes")
plt.ylabel("Views")
plt.title("YouTube Views Prediction")
plt.legend()
plt.show()