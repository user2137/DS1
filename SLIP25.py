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
from sklearn.preprocessing import LabelEncoder

df1 = pd.DataFrame({
    "Brand": ["Samsung", "Apple", "Redmi", "Vivo", "OnePlus"],
    "Model": ["S24", "iPhone15", "Note13", "V30", "Nord4"],
    "Price": [80000, 75000, 18000, 30000, 35000],
    "RAM": [8, 6, 6, 8, 12]
})

df2 = pd.DataFrame({
    "Model": ["S24", "iPhone15", "Note13", "V30", "Nord4"],
    "Storage": [256, 128, 128, 256, 256],
    "Battery_Capacity": [4000, 3349, 5000, 5000, 5500],
    "Operating_System": [
        "Android", "iOS", "Android", "Android", "Android"
    ]
})

print("DataFrame 1:")
print(df1)

print("\nDataFrame 2:")
print(df2)

# a) Data Integration
merged_df = pd.merge(df1, df2, on="Model")

print("\nMerged DataFrame:")
print(merged_df)

# b) Label Encoding on Brand
label_encoder = LabelEncoder()
merged_df["Brand_Encoded"] = label_encoder.fit_transform(
    merged_df["Brand"]
)

# c) One-Hot Encoding on Operating_System
transformed_df = pd.get_dummies(
    merged_df,
    columns=["Operating_System"],
    dtype=int
)

print("\nTransformed DataFrame:")
print(transformed_df)