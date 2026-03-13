import pandas as pd

df = pd.read_csv("./data/Emotion_Bangla_dataset.csv")

print(df.head())
print(df['label'].value_counts())