##q.1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create Student Dataset
data = {
    "Study Hours": [2, 3, 4, 5, 6, 7, 8, 2, 5, 9],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 55, 78, 95],
    "Assignment Score": [40, 45, 50, 60, 65, 75, 85, 35, 62, 95],
    "Exam Score": [35, 40, 50, 55, 65, 75, 80, 30, 60, 95]
}

df = pd.DataFrame(data)

# Save Dataset
df.to_csv("student_performance.csv", index=False)

print("Student Performance Data:")
print(df)

# Calculate Correlation Matrix
correlation_matrix = df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))

# Generate Heat Map
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    vmin=-1,
    vmax=1
)

plt.title("Correlation Heat Map of Student Performance")
plt.show()

#q.2

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

data = {
    "Temperature": [20, 25, 30, 35, 40],
    "Humidity": [40, 50, 60, 70, 80],
    "WindSpeed": [5, 10, 15, 20, 25],
    "Pressure": [1000, 1005, 1010, 1015, 1020],
    "Rainfall": [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# a) Min-Max Scaling
minmax = MinMaxScaler()
minmax_data = pd.DataFrame(
    minmax.fit_transform(df),
    columns=df.columns
)

print("\nMin-Max Scaling:")
print(minmax_data.round(2))

# b) Standardization
standard = StandardScaler()
standard_data = pd.DataFrame(
    standard.fit_transform(df),
    columns=df.columns
)

print("\nStandardization:")
print(standard_data.round(2))

# c) Normalization
normalizer = Normalizer()
normalized_data = pd.DataFrame(
    normalizer.fit_transform(df),
    columns=df.columns
)

print("\nNormalization:")
print(normalized_data.round(4))