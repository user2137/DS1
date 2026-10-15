##q.1

import matplotlib.pyplot as plt

transport = ["Train", "Bicycle", "Bike", "Car", "Bus"]
employees = [220, 60, 70, 80, 40]

plt.figure(figsize=(8, 6))

plt.pie(
    employees,
    labels=transport,
    autopct="%1.2f%%",
    colors=["skyblue", "lightgreen", "orange", "pink", "yellow"],
    startangle=90,
    shadow=True
)

plt.title("Preferred Mode of Transportation Among Employees")
plt.show()


##q.2

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["Video Lecture", "Quiz"],
    ["Video Lecture", "Assignment"],
    ["Quiz", "Discussion Forum"],
    ["Video Lecture", "Quiz", "Assignment"],
    ["Assignment", "Discussion Forum"],
    ["Video Lecture", "Certificate"],
    ["Quiz", "Certificate"],
    ["Video Lecture", "Quiz"],
    ["Assignment", "Certificate"],
    ["Video Lecture", "Quiz", "Certificate"]
]

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_array, columns=te.columns_)

print("Learning Resources Dataset:")
print(df)

for support in [0.3, 0.4]:
    print("\nMinimum Support:", support)

    frequent_itemsets = apriori(
        df,
        min_support=support,
        use_colnames=True
    )

    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    if len(frequent_itemsets) > 0:
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=0.1
        )

        print("\nAssociation Rules:")

        if not rules.empty:
            print(rules[
                ["antecedents", "consequents",
                 "support", "confidence", "lift"]
            ])
        else:
            print("No association rules found.")