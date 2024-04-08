# 获取词典

from Public.path import path_vocab
import pandas as pd
import re
# from nltk import word_tokenize

import json

unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'


# 获取 word to index 词典
def get_w2i(vocab_path=path_vocab):
    w2i = {}
    # with open(vocab_path, 'r', encoding="utf-8") as f:
    #      w2i = json.load(f)
    # w2i = {k: (v + 3) for k, v in w2i.items()}
    #
    # # w2i["[PAD]"] = 0
    # # w2i["[CLS]"] = 1
    # # w2i["[UNK]"] = 2  # unknown
    # # w2i["[SEP]"] = 3

    # with open(vocab_path, 'r', encoding="utf-8") as f:
    #     w2i = f.readlines()
    #
    # w2i = {v.replace("\n", ""): index for index, v in enumerate(w2i)}
    with open("./tokenizer_sms_all.json", "r", encoding="utf-8") as wf:
        tokenizer_sms_all = json.load(wf)
    return w2i




path = "/Volumes/junge/project/fengkong"


def get_vocab_idcnn():
    train_data_1 = pd.read_csv(path + '/sms_data_t.csv', names=['opposite_phone', 'content', 'un1',
                                                                'time1', 'time2', 'un2', 'un3', 'bkg', 'uid'])
    print("data_1 shape:", train_data_1.shape)
    train_data_2 = pd.read_csv(path + '/sms_data_t_2.csv', names=['opposite_phone', 'content', 'un1',
                                                                  'time1', 'time2', 'un2', 'un3',
                                                                  'bkg', 'uid'
                                                                  ])
    print("train_data_2 shape:", train_data_2.shape)

    train_data_we_1 = pd.read_csv(path + '/sms_data_m.csv', names=['opposite_phone', 'content', 'un1',
                                                                   'time1', 'time2', 'un2', 'un3', 'bkg', 'uid'])
    print("train_data_we_1 shape:", train_data_we_1.shape)

    stop_words = {'', 'no', 'x', 'hers', "she's", "aren't", 'ourselves', 'c', 'between', 'over', 'b', "couldn't",
                  'you',
                  've', 'don', 'm', 'them', 'which', 'does', 'more', 'needn', 'it', 'mustn', 'on', 'shan', 'such', 'g',
                  'because', 'each', 'nor', 'the', "didn't", 'up', 'but', 'from', 'and', "hasn't", 'or', "you'd", 's',
                  'to',
                  "haven't", 'not', 'am', 'https', 'wasn', 'those', 'xxxxx', "shouldn't", 'under', 'by', 'themselves',
                  'while', 'should', 'other', 'yourself', 'didn', 't', 'further', 'xxx', 'u', 'down', 'won',
                  'xxxxxxxxx',
                  'she', 'all', "doesn't", 'very', "mustn't", 'most', 'shouldn', "wouldn't", 'that', 'again', 'he',
                  're',
                  'will', 'z', "don't", 'p', 'ac', 'isn', 'were', 'h', 'y', 'this', 'through', 'an', 'me', 'any', 'k',
                  'being', 'than', 'at', 'i', 'him', 'q', 'their', 'd', 'against', 'v', 'now', 'its', 'been', 'my',
                  'these',
                  'ma', "needn't", 'above', 'xxxx', 'before', 'com', "mightn't", 'own', 'as', 'has', "wasn't", 'just',
                  'her', 'how', 'we', 'f', 'into', 'once', 'did', 'o', 'of', 'is', 'myself', 'wouldn', 'few', "isn't",
                  'where', 'be', 'yourselves', 'after', 'who', 'couldn', 'mightn', 'too', "shan't", 'e', 'same', 'his',
                  "it's", 'our', 'both', 'xx', 'ain', 'weren', 'if', 'until', 'yours', "you've", 'your', 'have',
                  "hadn't",
                  'only', 'theirs', "won't", 'off', 'some', 'having', "you're", 'whom', 'for', 'haven', 'a', 'with',
                  'aren',
                  'below', 'll', "should've", 'hasn', 'hadn', 'then', 'when', 'do', 'herself', 'had', 'j', 'was',
                  'are',
                  'what', 'here', 'doesn', 'why', 'himself', 'ours', 'can', "weren't", 'in', 'itself', "that'll", '\n',
                  'doing', 'during', 'out', 'there', 'they', 'w', 'n', 'r', 'l', 'so', 'about', "you'll"}

    train_all_data_t = pd.concat([train_data_1, train_data_2], axis=0)
    train_all_data_t = pd.concat([train_all_data_t, train_data_we_1], axis=0)

    all_tokens = []
    all_tokens.append(cls_flag)
    all_tokens.append(pad_flag)
    all_tokens.append(sep_flag)
    all_tokens.append(pad_flag)

    def clean_data(txt):
        txt = txt.replace('\n', ' ')
        txt = re.sub(r"([.,!:?()''])", r" \1 ", txt)
        txt = re.sub(r"\s{2,}", " ", txt)  # 将空白、换行符等替换
        txt = re.sub('[0-9]', ' ', txt)  # 去数字替换为x
        txt = txt.lower()  # 统一小写
        txt = re.sub('[^a-zA-Z]', ' ', txt)  # 去除非英文字符并替换为空格
        txt = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', ' ', txt)

        word_tokens = word_tokenize(txt)  # 分词
        filtered_word = [w for w in word_tokens if not w in stop_words]
        all_tokens.extend(filtered_word)

    train_all_data_t['content'].apply(lambda x: clean_data(txt=str(x)))

    dict_s = {}
    i = 0
    all_tokens = all_tokens.extend(list(get_w2i(vocab_path=path_vocab).kyes()))
    all_tokens = set(all_tokens)

    for one in all_tokens:
        dict_s[one] = i
        i += 1

    json_str = json.dumps(dict_s)
    with open(r"/Users/zhuyingjun/Desktop/fengkong/fengkong_model/ner_demo/data/vocab/en_char2id.json", "w") as wf:
        wf.write(json_str)


if __name__ == '__main__':
    import tensorflow as tf

    # dicr= {'1333':23,"2333":34}
    # print(type(list(dicr.keys())))
    # print(list(dicr.keys()))
    # json_str = json.dumps(dicr)
    # with open(r"/Users/zhuyingjun/Desktop/fengkong/fengkong_model/ner_demo/data/vocab/en_char2id.json","r") as wf :
    #     w=  json.load(wf)
    #     print(w.get("12","hhh"))
    #     print(type(w),w)
    # lists = [1,2,3,4,51,1]
    # print(type(set(lists)))
    # for  inem in set(lists):
    #     print(inem)
    # get_vocab_idcnn()
    # 得到分类的独热向量
    # targets = tf.one_hot(1, 2)
    # print(str(targets))
    # from keras.datasets import imdb
    #
    # # 这里使用10000个常用单词，减少计算
    # (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    # print(train_labels)
    # print(train_data[0])
    # print(type(train_data))
    get_w2i()
