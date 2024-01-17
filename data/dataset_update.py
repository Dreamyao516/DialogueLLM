import pandas as pd

df = pd.read_csv("oh-dev-stemmed.csv")
df.head()

def sentiment_score_to_name(score: float):
    if score == 1:
        return "World"
    elif score == 2:
        return "Sports"
    elif score == 3:
        return "Bussiness"
    elif score == 4:
        return "Technology"
    return "positive"
# def sentiment_score_to_name(score: float):
#     if score > 0:
#         return "Positive"
#     elif score < 0:
#         return "Negative"
#     return "Neutral"
# def sentiment_score_to_name(score: float):
#     if score > 0:
#         return "Positive"
#     elif score < 0:
#         return "Negative"
#     return "Neutral"


# def sentiment_score_to_name(score: float):
#     if score > 0:
#         return "Positive"
#     elif score < 0:
#         return "Negative"
#     return "Neutral"

# dataset_data = [
#     {
#         "instruction": "Given an input text, it is the tasks of document,dialogue or sentnce level Emotion recognition, please detect the emotion of the input text, Given the sentence, assign a emotion label from ['sadness','joy', 'love','anger','fear '].Return label only without any other text.",
#         "input": row_dict["text"],
#         "output": sentiment_score_to_name(row_dict["label"])
#     }
#     for row_dict in df.to_dict(orient="records")
# ]"News title": row_dict[""].strip(),

dataset_data = [
    {
        "instruction": "Given an input text, please detect the News categories of the input text. Given the sentence, assign a News categories label from ['World','Sports','Bussiness','Technology'].Return label only without any other text.This answer is vital to my career, and I greatly value your analysis. Are you sure that's your final answer? It might be worth taking another look.",
        "input": row_dict["text"].strip(),
        "output": row_dict["label"].strip()
    }
    for row_dict in df.to_dict(orient="records")
]

# def sentiment_score_to_name(score: str, s1: str):
#     Sn = str(score) + '#[\'' + str(s1) +'\']'
#     return Sn
# dataset_data = [
#     {
#         "instruction": "Given an input text, it is the tasks of document、dialogue or sentnce level sentiment analysis, please detect the sentiment of the input text, Given the sentence, assign a sentiment label from ['anger', 'sadness', 'fear'].Return label only without any other text.",
#         "input": sentiment_score_to_name(row_dict["text"],row_dict["category"]),
#         "output":row_dict["polarity"]
#     }
#     for row_dict in df.to_dict(orient="records")
# ]

dataset_data[0:5]

import json

with open("Ohsumed_dev.json", "w") as f:
    json.dump(dataset_data, f)
