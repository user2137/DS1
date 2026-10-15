#q.1

import pandas as pd

data = {
    "Person_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [18, 22, 25, 28, 30, 35, 40, 45, 50, 60]
}

age_records = pd.DataFrame(data)

print("Original Data:")
print(age_records)

age_records["Age_Bin"] = pd.qcut(
    age_records["Age"],
    q=3,
    labels=["Young", "Middle", "Old"]
)

print("\nDiscretized Data:")
print(age_records)

##q.2

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

data = {
    "City": ["Mumbai", "Pune", "Nashik", "Nagpur", "Delhi"],
    "Area (sq. km)": [603, 516, 259, 228, 1484],
    "Average Temperature (C)": [27, 25, 24, 29, 26],
    "Rainfall (mm)": [2200, 800, 700, 1100, 800],
    "Population Density": [20000, 5500, 6000, 4800, 12000]
}

df = pd.DataFrame(data)

df.to_csv("cities.csv", index=False)

print("City Dataset:")
print(df)

fig1 = px.treemap(
    df,
    path=["City"],
    values="Population Density",
    color="Population Density",
    color_continuous_scale="Viridis",
    title="Population Density of Different Cities"
)

fig1.show()

fig2 = px.scatter_3d(
    df,
    x="Area (sq. km)",
    y="Average Temperature (C)",
    z="Rainfall (mm)",
    color="City",
    size="Population Density",
    hover_name="City",
    title="3D Scatter Plot of City Data",
    labels={
        "Area (sq. km)": "Area (sq. km)",
        "Average Temperature (C)": "Average Temperature (°C)",
        "Rainfall (mm)": "Rainfall (mm)"
    }
)

fig2.show()