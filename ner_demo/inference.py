import tensorflow as tf
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
from DataProcess.vocab import *
import pandas as pd
import re  # 正则匹配
from collections import Counter

w2i = get_w2i()  # word to index
max_len = 200
unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'
import os
current_path = os.path.dirname(os.path.abspath(__file__))
unk_index = w2i.get(unk_flag, 2)
pad_index = w2i.get(pad_flag, 0)

stop_word = {'', 'no', 'x', 'hers', "she's", "aren't", 'ourselves', 'c', 'between', 'over', 'b', "couldn't", 'you',
             've', 'don', 'm', 'them', 'which', 'does', 'more', 'needn', 'it', 'mustn', 'on', 'shan', 'such', 'g',
             'because', 'each', 'nor', 'the', "didn't", 'up', 'but', 'from', 'and', "hasn't", 'or', "you'd", 's', 'to',
             "haven't", 'not', 'am', 'https', 'wasn', 'those', 'xxxxx', "shouldn't", 'under', 'by', 'themselves',
             'while', 'should', 'other', 'yourself', 'didn', 't', 'further', 'xxx', 'u', 'down', 'won', 'xxxxxxxxx',
             'she', 'all', "doesn't", 'very', "mustn't", 'most', 'shouldn', "wouldn't", 'that', 'again', 'he', 're',
             'will', 'z', "don't", 'p', 'ac', 'isn', 'were', 'h', 'y', 'this', 'through', 'an', 'me', 'any', 'k',
             'being', 'than', 'at', 'i', 'him', 'q', 'their', 'd', 'against', 'v', 'now', 'its', 'been', 'my', 'these',
             'ma', "needn't", 'above', 'xxxx', 'before', 'com', "mightn't", 'own', 'as', 'has', "wasn't", 'just',
             'her', 'how', 'we', 'f', 'into', 'once', 'did', 'o', 'of', 'is', 'myself', 'wouldn', 'few', "isn't",
             'where', 'be', 'yourselves', 'after', 'who', 'couldn', 'mightn', 'too', "shan't", 'e', 'same', 'his',
             "it's", 'our', 'both', 'xx', 'ain', 'weren', 'if', 'until', 'yours', "you've", 'your', 'have', "hadn't",
             'only', 'theirs', "won't", 'off', 'some', 'having', "you're", 'whom', 'for', 'haven', 'a', 'with', 'aren',
             'below', 'll', "should've", 'hasn', 'hadn', 'then', 'when', 'do', 'herself', 'had', 'j', 'was', 'are',
             'what', 'here', 'doesn', 'why', 'himself', 'ours', 'can', "weren't", 'in', 'itself', "that'll", '\n',
             'doing', 'during', 'out', 'there', 'they', 'w', 'n', 'r', 'l', 'so', 'about', "you'll"}


def data_process(text):
     '''
     text : 用户所有的短信数据拼接成了一条文本 ，具体的拼接方式看清洗数据的脚本：process_token200_data_11.py 下  get_hebing() 函数
     return 返回词频 top200
     '''

     txt = re.sub(r"([.,!:?()''<>])", r"", text)
     txt = re.sub(r"\s{2,}", " ", txt)  # 将空白、换行符等替换
     txt = re.sub('[0-9]', '', txt)  # 去数字替换为x
     txt = txt.lower()  # 、统一小写
     txt = re.sub('[^a-zA-Z]', ' ', txt)  # 去除非英文字符并替换为空格
     txt = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', txt)

     word_tokens = word_tokenize(txt)  # 分词
     # 这里没有使用 nltk 自带的 stop_word

     filtered_word = [w for w in word_tokens if not w in stop_word]
     c = Counter(filtered_word)
     top_120 = c.most_common(200)
     top_200 = [item[0] for item in top_120]
     return ' '.join(top_200)

def tokensbilstmid(top_tokens):
     data, line_data = [], []
     for w in top_tokens.split(" "):
          char_index = w2i.get(w, w2i[unk_flag])
          line_data.append(char_index)

     if len(line_data) < max_len:
          pad_num = max_len - len(line_data)
          line_data = line_data + [pad_index] * pad_num
     else:
          line_data = line_data[:max_len]
     data.append(line_data)

     return np.array(data)

from transformers import BertTokenizer

dict_path = r'/Users/zhuyingjun/Desktop/fengkong/ner_demo/data/data/uncased_L-12_H-768_A-12/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(dict_path)


def tokensbertid (top_tokens):
    max_len_buff = 198
    data_ids = []
    data_types = []
    label_ids = []

    row = top_tokens
    # bert 需要输入index和types 由于我们这边都是只有一句的，所以type都为0
    token_ids = tokenizer.encode(row)

    # 处理填充开始和结尾 bert 输入语句每个开始需要填充[CLS] 结束[SEP]
    if len(token_ids) >= max_len_buff:  # 先进行截断
        token_ids = token_ids[:max_len_buff]
        token_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]

    # padding
    else:  # 填充到最大长度
        pad_num = max_len_buff - len(token_ids)
        token_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id] + [
            tokenizer.pad_token_id] * pad_num

    seg_ids = [tokenizer.pad_token_id] * 200
    # print(token_ids)
    # print(f"tokenids len :{len(token_ids)}, seg_ids len :{len(seg_ids)}")
    assert len(token_ids) == len(seg_ids)
    data_ids.append(token_ids)
    data_types.append(seg_ids)

    return [np.array(data_ids), np.array(data_types)]

# from keras_bert import get_custom_objects
# model_bert = tf.keras.models.load_model(r"./save_model/BERT/1200/0.h5",custom_objects=get_custom_objects())

# model= tf.keras.models.load_model(r"./save_model/BILSTM/1.6w_200/50.h5")

top_words = " hi ni hao ya"

def get_sim_socre(top_words):
    result = model_bert.predict(tokensbertid(top_tokens=top_words))  # [[0.41026294 0.58973706]]  --->  [[0,1]]
    return result[0][1]
# print(get_sim_socre(top_words))

sim_path = "./bilst_fasttext_bert_socre.csv"
data_= pd.read_csv(sim_path)
# data_["Bilstm_socre"] = data_["tokens"].apply(lambda x: get_sim_socre(x))
# data_.to_csv("./bistm_fasttext_socre.csv")

print('pearson:', data_['Bilstm_socre'].corr(data_['BERT_socre']))
print('spearman', data_['Bilstm_socre'].corr(data_['BERT_socre'], method='spearman'))

