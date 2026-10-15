##q.1

import pandas as pd
data = {
    "Emp_ID": [101, 102, 103, 104, 105, 106, 106, 108],
    "Age": [25, 30, None, 40, 28, 35, 35, 100],
    "Dept": ["IT", "HR", "IT", None, "Sales", "HR", "HR", "IT"],
    "Years_Experience": [2, 5, 3, 15, None, 8, 8, 40],
    "Performance_Rating": [4, 3, 5, 4, 3, 4, 4, 5]
}

employee_data = pd.DataFrame(data)

print("Original Employee Data:")
print(employee_data)

employee_data["Age"] = employee_data["Age"].fillna(
    employee_data["Age"].median()
)

employee_data["Dept"] = employee_data["Dept"].fillna(
    employee_data["Dept"].mode()[0]
)

employee_data["Years_Experience"] = employee_data[
    "Years_Experience"
].fillna(employee_data["Years_Experience"].median())

print("\nAfter Handling Missing Values:")
print(employee_data)

employee_data = employee_data.drop_duplicates()

print("\nAfter Removing Duplicates:")
print(employee_data)

columns = ["Age", "Years_Experience", "Performance_Rating"]

print("\nOutliers Detected Using IQR:")

for col in columns:
    Q1 = employee_data[col].quantile(0.25)
    Q3 = employee_data[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = employee_data[
        (employee_data[col] < lower_limit) |
        (employee_data[col] > upper_limit)
    ]

    print("\n", col, "Outliers:")
    print(outliers[["Emp_ID", col]])

print("\nFinal Cleaned Employee Data:")
print(employee_data)

##q.2

import pandas as pd
import plotly.express as px

data = {
    "city": ["Mumbai", "Delhi", "London", "New York", "Tokyo",
             "Paris", "Beijing", "Sydney"],
    "country": ["India", "India", "UK", "USA", "Japan",
                "France", "China", "Australia"],
    "latitude": [19.0760, 28.6139, 51.5072, 40.7128,
                 35.6762, 48.8566, 39.9042, -33.8688],
    "longitude": [72.8777, 77.2090, -0.1276, -74.0060,
                  139.6503, 2.3522, 116.4074, 151.2093],
    "population": [12400000, 16700000, 8900000, 8800000,
                   14000000, 2100000, 21500000, 5300000]
}

world_cities = pd.DataFrame(data)

world_cities.to_csv("world_cities.csv", index=False)

print("World Cities Dataset:")
print(world_cities)

# a) Geospatial Visualization
fig1 = px.scatter_geo(
    world_cities,
    lat="latitude",
    lon="longitude",
    hover_name="city",
    hover_data=["country", "population"],
    size="population",
    color="country",
    title="World Cities Population Map",
    projection="natural earth",
    size_max=40
)

fig1.show()

# b) Treemap
fig2 = px.treemap(
    world_cities,
    path=["country", "city"],
    values="population",
    color="population",
    color_continuous_scale="Viridis",
    title="Population Distribution of World Cities"
)

fig2.show()