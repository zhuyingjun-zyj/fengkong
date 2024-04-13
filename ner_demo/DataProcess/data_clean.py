import json
from tqdm import tqdm
from pandas import Series
import calendar
import nltk, re
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
from datetime import datetime

tk = WordPunctTokenizer()  # F:\mylearning\fengkong\stanford-postagger-full-2017-06-09\stanford-postagger-3.8.0.jar


# nltk.download('punkt')


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


def get_week_distime(date_str, sub_time):

    try:
        year, month, day = map(int, date_str.split(' ')[0].split('-'))
        weekday = calendar.weekday(year, month, day)
        weekday_str = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][weekday]

        start_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(sub_time, "%Y-%m-%d %H:%M:%S")
        # 计算两个日期之间的天数差
        dis_time = (end_date - start_date).days

        print(f'{date_str}是{weekday_str} ，，  {date_str} ==>> {sub_time} 时间差 {dis_time}')
        return weekday_str, dis_time

    except Exception as e:
        print("有报错 ：" ,e)
        return "Sin fecha", "Sin fecha"


def get_content_clean(x):
    # print(x)
    sub_time = x["f_public_last_submit_time"]
    contents = x["f_sms_data"]
    contents = eval(contents)
    # contents = json.loads(result["f_sms_data"])
    contentsss = ""
    for content_one in contents:
        content = content_one.get("content")
        oppositePhone = content_one.get("oppositePhone")
        smsTime = content_one.get("smsTime")
        type = content_one.get("type")

        weekday, dis_time = get_week_distime(smsTime, sub_time)
        contentsss = contentsss + str(content) + " , tiempo " + str(smsTime) + " tiempo de entrada  " + str(
            dis_time) + " días , semana " + str(weekday) + "\n"

        # print(contentsss)
    print("===================")
    # result["f_sms_data"] = contentsss
    return Series([contentsss])


def clean_content(origin_file, label_file, out_file):
    """
    清洗短信内容
    """
    df = pd.read_csv(origin_file, sep='/001XG')  # /001XG
    df.dropna(subset=["f_sms_data"], inplace=True)
    df["f_sms_data"] = df.apply(lambda x: get_content_clean(x), axis=1)

    df_label = pd.read_csv(label_file)

    df_merge = pd.merge(df, df_label, on="f_public_uid", how="inner")

    df_merge.to_csv(out_file, index=False, sep='\t', encoding='utf-8')
    print(df_merge.shape)


def fit_tokenizer(merge_file):
    df_train = pd.read_csv(merge_file, sep="\t")
    df_train.dropna(subset=["f_sms_data"], inplace=True)
    df_train["f_sms_data"] = df_train["f_sms_data"].str.lower()
    df_train = df_train["f_sms_data"].tolist()

    print(len(df_train))

    # df_train = [
    #     re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', item.lower())
    #     for item in df_train]

    words_tokens = []
    for item in tqdm(df_train):

        for line in item.split("\n")[:-1]:
            line = line.split(" , el tiempo ")

            # 去除网址
            txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                         line[0])

            # 去除短信内容的数据 保留日期和具体时间
            txt = re.sub('[0-9]', ' ', txt)
            if len(line) >= 2:
                line = txt + " , el tiempo " + line[1]
            else:
                line = txt
            geek_line = tk.tokenize(line)
            words_tokens.extend(list(set(geek_line)))

    words_tokens = list(set(words_tokens))
    # print(f"分词后结果：{words_tokens}")

    # word : index
    words_tokens = {j: i for i, j in enumerate(words_tokens)}

    with open("./tokenizer_sms_test_clean.json", "w", encoding="utf-8") as wf:
        json.dump(words_tokens, wf, indent=2, ensure_ascii=False)
    print(f"分词：{words_tokens}")
    print(f" 字典词量 ：{len(words_tokens)}")


if __name__ == '__main__':
    origin_file = r"F:\mylearning/data/data/data_sms_old.csv"
    train_merge_file = r"F:\mylearning/data/data/train_data_sms_old_add_días_merge.csv"
    train_label_file = r"../data/data/data_label_old.csv"

    test_origin_file = r"../data/data/data_sms_new.csv"
    test_clean_file = r"../data/data/data_sms_new_clean.csv"
    test_label_file = r"../data/data/data_label_new.csv"
    test_merge_file = r"../data/data/test_merge.csv"

    # data2json(origin_file, out_file)
    # pd.read_csv(origin_file, sep='/001XG')[:50].to_csv("F:\mylearning/data/data/data_sms_old_50.csv", sep='\t')

    clean_content(origin_file, train_label_file, train_merge_file)

    # fit_tokenizer(merge_file)
    # fit_tokenizer(train_merge_file)

    # with open("./tokenizer_sms_test_clean.json", "r", encoding="utf-8") as wf:
    #     sms_test_clean = json.load(wf)
    #     with open("./tokenizer_sms_train_clean_1.5W_.json", "r", encoding="utf-8") as wf2:
    #         sms_train_clean_15w_ = json.load(wf2)
    #         with open("./tokenizer_sms_train_clean_15000.json", "r", encoding="utf-8") as wf3:
    #             sms_train_clean_15w = json.load(wf3)
    #             all_info = {**sms_test_clean, **sms_train_clean_15w_}
    #             all_info = {**all_info, **sms_train_clean_15w}
    #
    #               # unknown
    #             all_info = {k: (v + 3) for k, v in all_info.items()}
    #             all_info["[PAD]"] = 0
    #             all_info["[END]"] = 1
    #             all_info["[UNK]"] = 2
    #
    #             with open("./tokenizer_sms_all.json", "w", encoding="utf-8") as wf:
    #                 json.dump(all_info, wf, indent=2, ensure_ascii=False)
    #             print(f"分词：{all_info}")
    #             print(f" 字典词量 ：{len(all_info)}")
