##q.1

import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    "Customer_ID": [101, 102, 103, 104, 105],
    "Gender": ["Male", "Female", "Male", "Female", "Male"],
    "City": ["Nashik", "Pune", "Mumbai", "Pune", "Nashik"],
    "Membership_Type": ["Gold", "Silver", "Platinum", "Gold", "Silver"]
}

customer_DataFrame = pd.DataFrame(data)

print("DataFrame Before Encoding:")
print(customer_DataFrame)

le = LabelEncoder()

customer_DataFrame["Gender"] = le.fit_transform(
    customer_DataFrame["Gender"]
)

customer_DataFrame["City"] = le.fit_transform(
    customer_DataFrame["City"]
)

customer_DataFrame["Membership_Type"] = le.fit_transform(
    customer_DataFrame["Membership_Type"]
)

print("\nDataFrame After Encoding:")
print(customer_DataFrame)

##q.2

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

data = {
    "Customer_ID": range(1, 11),
    "Netflix": ["Yes", "Yes", "Yes", "Yes", "No",
                "No", "Yes", "No", "Yes", "No"],
    "Amazon_Prime": ["No", "Yes", "No", "Yes", "Yes",
                     "Yes", "Yes", "No", "No", "No"]
}

df = pd.DataFrame(data)

print("Customer DataFrame:")
print(df)


netflix_only = len(df[(df["Netflix"] == "Yes") &
                      (df["Amazon_Prime"] == "No")])

amazon_only = len(df[(df["Netflix"] == "No") &
                     (df["Amazon_Prime"] == "Yes")])

both = len(df[(df["Netflix"] == "Yes") &
              (df["Amazon_Prime"] == "Yes")])

print("\nNetflix Only:", netflix_only)
print("Amazon Prime Only:", amazon_only)
print("Both Services:", both)


plt.figure(figsize=(6, 6))

venn2(subsets=(netflix_only, amazon_only, both),
      set_labels=("Netflix", "Amazon Prime"),
      set_colors=("red", "blue"),
      alpha=0.6)

plt.title("Netflix and Amazon Prime Customers")
plt.show()


labels = ["Netflix Only", "Amazon Prime Only", "Both Services"]
values = [netflix_only, amazon_only, both]
colors = ["red", "blue", "purple"]

plt.figure(figsize=(7, 6))

plt.pie(values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90)

plt.title("Customer Subscription Distribution")
plt.show()