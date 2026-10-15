##q.1

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    "Product_ID": [101, 102, 103, 104, 105],
    "Price": [100, 200, 300, 400, 500],
    "Quantity_Sold": [10, 20, 30, 40, 50],
    "Discount": [5, 10, 15, 20, 25],
    "Revenue": [950, 3600, 7650, 12800, 18750]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

print("\nTransformed DataFrame:")
print(scaled_df.round(2))

##q.2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

data = {
    "RAM": [2, 3, 4, 4, 6, 6, 8, 8, 12, 12, 16, 16],
    "Storage": [32, 32, 64, 128, 64, 128, 128, 256, 256, 512, 256, 512],
    "Battery": [3000, 3500, 4000, 4200, 4000, 4500,
                4500, 5000, 5000, 5000, 5500, 6000],
    "Camera": [8, 12, 13, 16, 16, 24, 32, 48, 50, 64, 108, 200],
    "Price_Category": [
        "Low", "Low", "Low", "Low",
        "Medium", "Medium", "Medium", "Medium",
        "High", "High", "High", "High"
    ]
}

df = pd.DataFrame(data)

print("Smartphone Dataset:")
print(df)

X = df[["RAM", "Storage", "Battery", "Camera"]]
y = df["Price_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Categories:")
print(y_test.values)

print("\nPredicted Categories:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

cm = confusion_matrix(
    y_test, y_pred,
    labels=["Low", "Medium", "High"]
)

print("\nAccuracy:", accuracy)
print("F1-Score:", f1)

print("\nConfusion Matrix:")
print(cm)