# sumには感情合計後のスコアの合計が返ってくる

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# list_exa = ["ナイス", "なんの成果も！！得られませんでした！！", "だめだ"]


def model(text_list):
    # 感情認識モデルの構築
    model = AutoModelForSequenceClassification.from_pretrained(
        "jarvisx17/japanese-sentiment-analysis"
    )
    tokenizer = AutoTokenizer.from_pretrained("jarvisx17/japanese-sentiment-analysis")
    model_score_list = []
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # 以下、chat参照
    max_tokens = 500
    for i in text_list:
        # 文を最大トークン数に切り詰める
        truncated_text = " ".join(
            tokenizer.tokenize(tokenizer.decode(tokenizer.encode(i))[:max_tokens])
        )

        # 感情分析のモデルに通す
        model_score = nlp(truncated_text)

        # 結果をリストに追加
        model_score_list.append(model_score)

    # for i in text_list:
    #     model_score = nlp(i)
    #     model_score_list.append(model_score)
    # 次元を一つ減らす flattened_listで[{},{},{}]の形
    flattened_list = [item[0] for item in model_score_list]

    # sumを計算する
    sum = 0

    for item in flattened_list:
        label = item["label"]
        score = item["score"]

        if label == "positive":
            sum += score
        elif label == "negative":
            sum -= score

    return sum


# score_box = model(list_exa)
# print(score_box)
