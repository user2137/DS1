##q.1

import matplotlib.pyplot as plt

courses = ["BCA", "BBA", "B.Com", "B.Sc", "BA"]
students = [60, 45, 75, 50, 35]

colors = ["red", "blue", "green", "orange", "purple"]

plt.bar(courses, students, color=colors)

plt.title("Number of Students Enrolled in Different Courses")
plt.xlabel("Courses")
plt.ylabel("Number of Students")

plt.show()

##q.2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [40, 45, 50, 60, 65, 70, 80, 85, 90, 95],
    "Marks": [20, 25, 35, 40, 50, 60, 70, 80, 90, 95],
    "Performance": [
        "Pass", "Pass", "Pass", "Average", "Average",
        "Average", "Excellent", "Excellent", "Excellent", "Excellent"
    ]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

X = df[["Study_Hours", "Attendance", "Marks"]]
y = df["Performance"]

imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Performance:")
print(encoder.inverse_transform(y_test))

print("\nPredicted Performance:")
print(encoder.inverse_transform(y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))