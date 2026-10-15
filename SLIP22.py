##q.1

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    "City": ["Nashik", "Pune", "Mumbai", "Nashik", "Pune"],
    "House_Type": ["1BHK", "2BHK", "3BHK", "2BHK", "1BHK"],
    "House_Area": [500, 800, 1200, 900, 600],
    "Number_of_Rooms": [1, 2, 3, 2, 1],
    "Monthly_Rent": [5000, 10000, 20000, 12000, 7000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

num_cols = ["House_Area", "Number_of_Rooms", "Monthly_Rent"]

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nNormalized Dataset:")
print(df.round(2))

##q.2

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

data = {
    "Customer_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Customer_Name": [
        "Amit", "Priya", "Rahul", "Sneha", "Neha",
        "Rohan", "Pooja", "Akash", "Sahil", "Anjali"
    ],
    "Netflix": [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "Amazon_Prime": [0, 1, 1, 1, 1, 0, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

print("Customer Dataset:")
print(df)

netflix = set(df[df["Netflix"] == 1]["Customer_ID"])
prime = set(df[df["Amazon_Prime"] == 1]["Customer_ID"])

only_netflix = netflix - prime
only_prime = prime - netflix
both = netflix & prime

print("\nOnly Netflix:", len(only_netflix))
print("Only Amazon Prime:", len(only_prime))
print("Both Services:", len(both))

plt.figure(figsize=(7, 6))

venn2(
    subsets=(len(only_netflix), len(only_prime), len(both)),
    set_labels=("Netflix", "Amazon Prime"),
    set_colors=("red", "blue"),
    alpha=0.6
)

plt.title("Netflix vs Amazon Prime Users")
plt.show()

labels = ["Only Netflix", "Only Amazon Prime", "Both Services"]
sizes = [len(only_netflix), len(only_prime), len(both)]
colors = ["red", "blue", "green"]

plt.figure(figsize=(7, 6))

plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Distribution of Streaming Service Users")
plt.axis("equal")
plt.show()