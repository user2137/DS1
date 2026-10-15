##Q.1

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

data = {
    "Price": [100, 200, 150, 500, 450],
    "Rating": [4.2, 4.5, 4.0, 4.8, 4.6],
    "Sales": [50, 80, 60, 200, 180]
}

df = pd.DataFrame(data)
print(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

link = linkage(scaled_data, method="ward")

plt.figure(figsize=(8, 5))
dendrogram(link, labels=["P1", "P2", "P3", "P4", "P5"])
plt.title("Product Dendrogram")
plt.xlabel("Products")
plt.ylabel("Distance")
plt.show()

##q.2

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    "Application_ID": range(1, 11),
    "Age": [22, 25, 30, 35, 40, 28, 45, 32, 50, 26],
    "Income": [20000, 25000, 40000, 50000, 60000,
               30000, 70000, 45000, 80000, 28000],
    "Credit_Score": [600, 620, 700, 720, 750, 650, 780, 710, 800, 630],
    "Loan_Amount": [50000, 60000, 80000, 90000, 100000,
                    55000, 120000, 85000, 130000, 65000],
    "Employment_Years": [1, 2, 5, 7, 10, 3, 12, 6, 15, 2],
    "Debt_Ratio": [0.5, 0.6, 0.3, 0.2, 0.2,
                   0.5, 0.1, 0.3, 0.1, 0.6],
    "Loan_Status": [
        "Rejected", "Rejected", "Approved", "Approved",
        "Approved", "Rejected", "Approved", "Approved",
        "Approved", "Rejected"
    ]
}

df = pd.DataFrame(data)
df.to_csv("Loan_application.csv", index=False)

X = df.drop(["Application_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"].map({"Rejected": 0, "Approved": 1})

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

print("Feature Importance:")
print(pd.Series(model.feature_importances_,
                index=X.columns).sort_values(ascending=False))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Reduced Data:")
print(pd.DataFrame(X_pca, columns=["PC1", "PC2"]))

print("\nExplained Variance:")
print(pca.explained_variance_ratio_)