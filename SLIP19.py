##q.1

import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Fuel_Efficiency": [15, 18, 20, 22, 60],
    "Engine_Power": [80, 100, 120, 140, 400],
    "Vehicle_Weight": [1000, 1200, 1400, 1600, 1800],
    "Engine_Size": [1000, 1200, 1500, 1800, 2000],
    "Acceleration": [12, 11, 10, 9, 8]
}

df = pd.DataFrame(data)

print("Vehicle Data:")
print(df)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.boxplot(df["Fuel_Efficiency"])
plt.title("Fuel Efficiency Box Plot")
plt.ylabel("Fuel Efficiency (km/l)")

plt.subplot(1, 2, 2)
plt.boxplot(df["Engine_Power"])
plt.title("Engine Power Box Plot")
plt.ylabel("Engine Power (HP)")

plt.tight_layout()
plt.show()

for col in ["Fuel_Efficiency", "Engine_Power"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print("\nOutliers in", col, ":")
    print(outliers[col].tolist())

 ##q.2

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

data = {
    "Temperature": [20, 25, 30, 35, 40],
    "Humidity": [40, 50, 60, 70, 80],
    "Wind Speed": [5, 10, 15, 20, 25],
    "Pressure": [1000, 1005, 1010, 1015, 1020],
    "Rainfall": [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# a) Min-Max Scaling
scaler1 = MinMaxScaler()
minmax_data = pd.DataFrame(
    scaler1.fit_transform(df),
    columns=df.columns
)

print("\nMin-Max Scaling:")
print(minmax_data.round(2))

# b) Standardization
scaler2 = StandardScaler()
standard_data = pd.DataFrame(
    scaler2.fit_transform(df),
    columns=df.columns
)

print("\nStandardization:")
print(standard_data.round(2))

# c) Normalization
scaler3 = Normalizer()
normalized_data = pd.DataFrame(
    scaler3.fit_transform(df),
    columns=df.columns
)

print("\nNormalization:")
print(normalized_data.round(4))