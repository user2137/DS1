##q.1

import pandas as pd

student_details = pd.DataFrame({
    "Student_ID": [101, 102, 103, 104, 105],
    "Name": ["Amit", "Rahul", "Priya", "Sneha", "Rohit"],
    "Age": [20, 21, 19, 20, 22]
})

academic_records = pd.DataFrame({
    "Student_ID": [101, 102, 103, 104, 105],
    "Course": ["BSc CS", "BSc CS", "BSc CS", "BSc CS", "BSc CS"],
    "Marks": [85, 78, 92, 88, 75]
})

print("Student Details:")
print(student_details)

print("\nAcademic Records:")
print(academic_records)

integrated_data = pd.merge(
    student_details,
    academic_records,
    on="Student_ID"
)

print("\nIntegrated Data:")
print(integrated_data)

##q.2

import pandas as pd
import plotly.express as px

data = {
    "City": ["Mumbai", "Pune", "Nashik", "Nagpur", "Delhi"],
    "Population": [12400000, 7000000, 1800000, 2900000, 16700000],
    "GDP": [310, 69, 15, 25, 293],
    "Literacy Rate": [89, 91, 90, 93, 88]
}

df = pd.DataFrame(data)

df.to_csv("city_data.csv", index=False)

print("City Dataset:")
print(df)

# a) Generate Treemap
fig1 = px.treemap(
    df,
    path=["City"],
    values="Population",
    color="Population",
    color_continuous_scale="Viridis",
    title="City-wise Population Distribution"
)

fig1.show()

# b) Create 3D Scatter Plot
fig2 = px.scatter_3d(
    df,
    x="Population",
    y="GDP",
    z="Literacy Rate",
    color="City",
    hover_name="City",
    title="3D Scatter Plot of City Data",
    labels={
        "Population": "Population",
        "GDP": "GDP (in billion USD)",
        "Literacy Rate": "Literacy Rate (%)"
    }
)

fig2.show()