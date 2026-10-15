##q.1

import pandas as pd
import matplotlib.pyplot as plt

data = {
    "PassengerId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Pclass": [3, 1, 3, 1, 3, 2, 1, 3, 2, 1],
    "Survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    "Age": [22, 38, 26, 35, 35, 28, 54, 2, 27, 14]
}

df = pd.DataFrame(data)

df.to_csv("titanic.csv", index=False)

df = pd.read_csv("titanic.csv")

print("Titanic Dataset:")
print(df)

plt.hist(
    df["Pclass"],
    bins=[0.5, 1.5, 2.5, 3.5],
    color="skyblue",
    edgecolor="black"
)

plt.xticks([1, 2, 3])
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.title("Number of Passengers in Each Class")
plt.show()

survival_counts = df["Survived"].value_counts().sort_index()

plt.pie(
    survival_counts,
    labels=["Not Survived", "Survived"],
    autopct="%1.1f%%",
    colors=["tomato", "lightgreen"],
    startangle=90
)

plt.title("Titanic Passenger Survival Distribution")
plt.show()

##q.2

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

data = {
    "word_freq_free": [0.5, 0.0, 0.8, 0.1, 0.0, 1.2, 0.2, 0.0, 0.9, 0.1,
                       0.0, 1.5, 0.3, 0.0, 0.7, 0.1, 0.0, 1.0, 0.2, 0.0],
    "word_freq_money": [0.4, 0.0, 0.6, 0.0, 0.1, 1.0, 0.0, 0.0, 0.8, 0.0,
                        0.0, 1.2, 0.1, 0.0, 0.5, 0.0, 0.0, 0.9, 0.0, 0.1],
    "word_freq_offer": [0.3, 0.0, 0.7, 0.1, 0.0, 0.9, 0.0, 0.0, 0.6, 0.0,
                        0.0, 1.1, 0.2, 0.0, 0.8, 0.0, 0.1, 0.7, 0.0, 0.0],
    "char_freq_dollar": [0.2, 0.0, 0.5, 0.0, 0.1, 0.8, 0.0, 0.0, 0.6, 0.0,
                         0.0, 0.9, 0.1, 0.0, 0.4, 0.0, 0.0, 0.7, 0.0, 0.1],
    "spam": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0,
             0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

df.to_csv("spambase.csv", index=False)

df = pd.read_csv("spambase.csv")

print("Spambase Dataset:")
print(df.head())

X = df.drop("spam", axis=1)
y = df["spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-Score:", f1_score(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))