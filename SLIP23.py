##q.1

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

data = {
    "Product_name": [
        "Laptop", "Mobile", "Tablet", "Headphones",
        "Smartwatch", "Keyboard", "Mouse", "Monitor"
    ],
    "Price": [50000, 20000, 30000, 2000, 5000, 1500, 800, 12000],
    "Rating": [4.5, 4.3, 4.2, 4.1, 4.0, 4.2, 4.0, 4.4],
    "Number_of_Sales": [100, 250, 150, 400, 300, 350, 500, 120]
}

df = pd.DataFrame(data)

print("Product Dataset:")
print(df)

X = df[["Price", "Rating", "Number_of_Sales"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 6))

dendrogram(
    linked,
    labels=df["Product_name"].values,
    leaf_rotation=45,
    leaf_font_size=10
)

plt.title("Dendrogram of Similar Products")
plt.xlabel("Product Name")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.show()

##q.2

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

data = {
    "Temperature": [25, 30, 35, 20, 28],
    "Humidity": [60, 70, 80, 50, 65],
    "Wind Speed": [10, 15, 20, 8, 12],
    "Pressure": [1010, 1015, 1020, 1005, 1012],
    "Rainfall": [20, 40, 60, 10, 30]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# A. Min-Max Scaling
minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(df)

minmax_df = pd.DataFrame(
    minmax_data,
    columns=df.columns
)

print("\nA. Min-Max Scaled Data:")
print(minmax_df.round(2))

# B. Standardization
standard = StandardScaler()
standard_data = standard.fit_transform(df)

standard_df = pd.DataFrame(
    standard_data,
    columns=df.columns
)

print("\nB. Standardized Data:")
print(standard_df.round(2))

# C. Normalization
normalizer = Normalizer()
normalized_data = normalizer.fit_transform(df)

normalized_df = pd.DataFrame(
    normalized_data,
    columns=df.columns
)

print("\nC. Normalized Data:")
print(normalized_df.round(4))