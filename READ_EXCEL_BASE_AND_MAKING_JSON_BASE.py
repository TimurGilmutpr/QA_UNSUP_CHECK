import pandas as pd 
import numpy as np
import json
import os


# читаем директорию
list_samples = [f"ПРИМЕРЫ/{k}" for k in os.listdir("ПРИМЕРЫ") if k.endswith(".xlsx") and "~" not in k]
list_samples


dfs = []
for k in list_samples:
    df = pd.read_excel(k)
    dfs.append(df)
dfs_all = pd.concat(dfs, ignore_index = True)
dfs_all.to_excel("ВСЕ.xlsx")
dfs_all


print(dfs_all["Класс"].value_counts().sum())

#dfs_all[dfs_all["Класс"].isna()].to_excel("ИРЕ_незаполненные.xlsx")

dfs_to_json = dfs_all[dfs_all["Класс"].notna()].reset_index()
dfs_to_json



from sklearn.model_selection import train_test_split
SYSTEM = (
    "Отвечай только на русском языке. "
    "Ты врач-генетик. Опиши, что это такое, чем грозит для женщин и мужчин, "
    "какие риски для потомства, шаги, которые нужно предпринять. Пиши чётко, но понятно, без метафор. "
    "Используй знания из ISCN 2020 и Gardner and Sutherland."
)



class_counts = dfs_to_json["Класс"].value_counts()

valid_classes = class_counts[class_counts > 2].index

dfs_to_json = dfs_to_json[dfs_to_json["Класс"].isin(valid_classes)]


train_df, val_df = train_test_split(dfs_to_json, test_size=0.1, random_state=42, shuffle = True, stratify = dfs_to_json["Класс"])



def to_record(row):
    user = f"Вопрос: {row['Вопрос']}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["Ответ"]},
        ]
    }

def dump_jsonl(df_part, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df_part.iterrows():
            f.write(json.dumps(to_record(row), ensure_ascii=False) + "\n")


dump_jsonl(train_df, "JSON_MARKED/train.jsonl")
dump_jsonl(val_df, "JSON_MARKED/val.jsonl")

print("train:", len(train_df), "val:", len(val_df))