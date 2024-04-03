import json
import pandas as pd
from pandas import Series
import calendar
import re
import nltk
from nltk import word_tokenize

nltk.download('punkt')


def data2json(origin_file, out_file):
    """
    安心提供的最原始的地址数据
    """
    df = pd.read_csv(origin_file, sep='/001XG')
    f_public_uid = ""
    results_final = {}
    for index, item in df.iterrows():
        if index >= 2:
            result = {"f_public_last_submit_time": None, "f_sms_data": None, "f_scrapy_data": None,
                      "f_public_loan_num": None, "f_public_uid": None, "f_app_data": None}
            keys = list(result.keys())
            # print(item)
            for index, key in enumerate(keys):
                result[key] = item[key]

            # result["f_sms_data"] = result["f_sms_data"].replace('oppositePhone','"oppositePhone"').replace('smsTime','"smsTime"').replace('type','"type"')
            result["f_sms_data"] = result["f_sms_data"]

            contents = eval(result["f_sms_data"])
            # contents = json.loads(result["f_sms_data"])
            contentsss = ""
            for content_one in contents:
                content = content_one["content"]
                oppositePhone = content_one.get("oppositePhone")
                smsTime = content_one.get("smsTime")
                type = content_one.get("type")
                weekday = get_week(smsTime)
                contentsss = contentsss + content + ", el tiempo:" + smsTime + ", semana :" + weekday + "\n"
            print("===================")
            result["f_sms_data"] = contentsss
            results_final[result["f_public_uid"]] = result["f_sms_data"]
    with open(out_file, "w", encoding="utf-8") as wf:
        json.dump(results_final, wf, ensure_ascii=False, indent=2)


def get_week(date_str):
    # print(date_str)
    try:
        year, month, day = map(int, date_str.split(' ')[0].split('-'))
        weekday = calendar.weekday(year, month, day)
        weekday_str = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][weekday]
        print(f'{date_str}是{weekday_str}')
        return weekday_str
    except:
        return "Sin fecha"


def get_content_clean(contents):
    contents = eval(contents)
    # contents = json.loads(result["f_sms_data"])
    contentsss = ""
    for content_one in contents:
        content = content_one["content"]
        oppositePhone = content_one.get("oppositePhone")
        smsTime = content_one.get("smsTime")
        type = content_one.get("type")
        weekday = get_week(smsTime)
        contentsss = contentsss + content + " , el tiempo " + smsTime + " , semana " + weekday + "\n"
        # print(contentsss)
    print("===================")
    # result["f_sms_data"] = contentsss
    return Series([contentsss])


def clean_content(origin_file, out_file):
    """
    清洗短信内容
    """
    df = pd.read_csv(origin_file, sep='/001XG')
    df["f_sms_data"] = df["f_sms_data"].apply(lambda x: get_content_clean(x))
    df.to_csv(out_file, index=False, sep='\t', encoding='utf-8')
    print(df.shape)


def merge_sms_label(content_file, label_file, merge_file):
    '''

    '''

    df_content = pd.read_csv(content_file, sep="\t")
    df_label = pd.read_csv(label_file)
    # df_label = df_label[["f_public_uid","apply_id",]]
    df_merge = pd.merge(df_content, df_label, on="f_public_uid", how="inner")
    print(df_merge.shape)
    df_merge.to_csv(merge_file, sep="\t", index=False)
    return df_merge


def fit_tokenizer(merge_file):
    df_content = pd.read_csv(merge_file, sep="\t")
    df_content.dropna(subset=["f_sms_data"], inplace=True)
    df_train = df_content["f_sms_data"].tolist()

    print(len(df_train))

    df_train = [
        re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', item.lower())
        for item in df_train]

    words_tokens = []
    for item in df_train:

        word_tokens = word_tokenize(item)

        for item_ in word_tokens:
            result = re.search(r'(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d', item_)
            print(f"result ： {result}")
            if result:
                result = result[0].split(":")
                words_tokens.extend(list(set(result)))
            else:
                if len(item_.split("-")) >= 2:
                    result = item_.split("-")
                    words_tokens.extend(result)

                if len(item_.split("/")) >= 2:
                    result = item_.split("/")
                    words_tokens.extend(result)
                else:
                    words_tokens.append(item_)

    print(f"分词后结果：{words_tokens}")
    words_tokens = list(set(words_tokens))

    # word : index
    words_tokens = {j: i for i, j in enumerate(words_tokens)}

    with open("./tokenizer_sms_test.json", "w", encoding="utf-8") as wf:
        json.dump(words_tokens, wf, indent=2, ensure_ascii=False)
    print(f"分词：{words_tokens}")
    print(f"字典词量：{len(words_tokens)}")


if __name__ == '__main__':
    origin_file = r"../data/data/data_sms_old.csv"
    train_clean_file = r"../data/data/data_sms_old_clean.csv"
    train_label_file = r"../data/data/data_label_old.csv"
    train_merge_file = r"../data/data/train_merge.csv"

    test_origin_file = r"../data/data/data_sms_new.csv"
    test_clean_file = r"../data/data/data_sms_new_clean.csv"
    test_label_file = r"../data/data/data_label_new.csv"
    test_merge_file = r"../data/data/test_merge.csv"

    # data2json(origin_file, out_file)
    clean_content(test_origin_file, test_clean_file)
    # merge_sms_label(train_clean_file, train_label_file, merge_file)
    # fit_tokenizer(merge_file)
