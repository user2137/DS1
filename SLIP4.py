##q.1

import matplotlib.pyplot as plt

payment_methods = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking"]
customers = [150, 80, 60, 40, 30]

plt.pie(customers, labels=payment_methods, autopct="%1.1f%%",
        startangle=90)

plt.title("Preferred Payment Methods Among Customers")
plt.show()

##q.2

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    "Annual_Income": [15, 18, 20, 22, 25, 50, 55, 60, 65, 70],
    "Spending_Score": [80, 85, 75, 90, 88, 40, 45, 35, 30, 42],
    "Purchase_Frequency": [12, 15, 10, 14, 13, 6, 7, 5, 4, 6]
}

df = pd.DataFrame(data)

print("E-commerce Customer Dataset:")
print(df)


print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


X = df[["Annual_Income", "Spending_Score",
        "Purchase_Frequency"]]


imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCustomer Clusters:")
print(df)


print("\nCluster Analysis:")
print(df.groupby("Cluster")[
    ["Annual_Income", "Spending_Score",
     "Purchase_Frequency"]
].mean().round(2))


plt.scatter(df["Annual_Income"], df["Spending_Score"],
            c=df["Cluster"], cmap="viridis", s=100)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("E-commerce Customer Segmentation")
plt.show()