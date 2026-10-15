##q.1

import pandas as pd

data = {
    "Book_ID": [101, 102, 103, 104, 105,
                106, 107, 108, 109, 110],
    "Book_Price": [100, 150, 200, 250, 300,
                   350, 400, 450, 500, 550]
}

book_records = pd.DataFrame(data)

print("Original Data:")
print(book_records)

book_records["Price_Category"] = pd.cut(
    book_records["Book_Price"],
    bins=3,
    labels=["Low", "Medium", "High"]
)

print("\nDiscretized Data:")
print(book_records)

##q.2

import pandas as pd
import plotly.express as px

data = {
    "city": [
        "Mumbai", "Delhi", "London", "New York",
        "Tokyo", "Paris", "Sydney", "Beijing",
        "Dubai", "Singapore"
    ],
    "country": [
        "India", "India", "United Kingdom", "USA",
        "Japan", "France", "Australia", "China",
        "UAE", "Singapore"
    ],
    "latitude": [
        19.0760, 28.6139, 51.5072, 40.7128,
        35.6762, 48.8566, -33.8688, 39.9042,
        25.2048, 1.3521
    ],
    "longitude": [
        72.8777, 77.2090, -0.1276, -74.0060,
        139.6503, 2.3522, 151.2093, 116.4074,
        55.2708, 103.8198
    ],
    "population": [
        12400000, 16700000, 8900000, 8800000,
        13900000, 2100000, 5300000, 21500000,
        3500000, 5700000
    ]
}

df = pd.DataFrame(data)

print("World Cities Dataset:")
print(df)

# A. Geospatial Visualization
fig1 = px.scatter_geo(
    df,
    lat="latitude",
    lon="longitude",
    locations=None,
    hover_name="city",
    hover_data=["country", "population"],
    size="population",
    color="country",
    projection="natural earth",
    title="Geospatial Visualization of World Cities"
)

fig1.show()

# B. Treemap
fig2 = px.treemap(
    df,
    path=["country", "city"],
    values="population",
    color="population",
    color_continuous_scale="Viridis",
    title="Population Distribution of World Cities"
)

fig2.show()
