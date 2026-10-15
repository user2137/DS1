##q.1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "Temperature": [25, 28, 30, 22, 24, 32, 27, 20],
    "Humidity": [60, 65, 70, 80, 75, 55, 68, 85],
    "Rainfall": [10, 15, 20, 40, 35, 5, 18, 50],
    "Wind Speed": [12, 15, 18, 10, 11, 20, 14, 8]
}

df = pd.DataFrame(data)

print("Weather Dataset:")
print(df)

correlation = df.corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Heat Map of Weather Parameters")
plt.show()

##q.2

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = {
    "Student": ["S1", "S2", "S3", "S4", "S5",
                "S6", "S7", "S8", "S9", "S10"],
    "Attendance": [90, 85, 80, 70, 65, 60, 50, 45, 40, 30],
    "Marks": [92, 88, 85, 75, 70, 65, 55, 50, 45, 35]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

X = df[["Attendance", "Marks"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

centroids = scaler.inverse_transform(kmeans.cluster_centers_)

print("\nStudent Cluster Assignments:")
print(df)

print("\nCluster Centroids:")
print(pd.DataFrame(
    centroids,
    columns=["Attendance", "Marks"]
).round(2))

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Attendance"],
    df["Marks"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    c="red",
    label="Centroids"
)

for i in range(len(df)):
    plt.annotate(
        df["Student"][i],
        (df["Attendance"][i], df["Marks"][i]),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("Attendance (%)")
plt.ylabel("Marks")
plt.title("K-Means Clustering of Students (K = 3)")
plt.legend()
plt.grid(True)
plt.show()