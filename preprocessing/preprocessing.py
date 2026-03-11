import pandas as pd
import re

def clean_text(text):

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    text = re.sub(r"[^\u0980-\u09FF\s]", "", text)

    text = text.strip()

    return text


df = pd.read_csv("data/Emotion_Bangla_dataset.csv")

df["clean_text"] = df["text"].apply(clean_text)

df.to_csv("data/clean_dataset.csv", index=False)

print("Preprocessing complete")