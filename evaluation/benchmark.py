import pandas as pd

data = {
    "Model": ["Logistic Regression", "SVM", "BanglaBERT"],
    "Accuracy": [0.72, 0.75, 0.86],
    "F1": [0.70, 0.74, 0.85]
}

df = pd.DataFrame(data)

df.to_csv("./data/metrics.csv", index=False)

print(df)