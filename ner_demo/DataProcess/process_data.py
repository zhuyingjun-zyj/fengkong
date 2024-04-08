from DataProcess.vocab import *
from Public.path import path_data_dir, path_data2_dir, path_msra_dir, path_renmin_dir
import numpy as np, pandas as pd
import os, re

import codecs
from nltk import word_tokenize
from bert4keras.tokenizers import Tokenizer
from Public.path import path_vocab
from nltk.tokenize import WordPunctTokenizer


class DataProcess(object):
    def __init__(self, max_len=12800, model=None):
        """
        数据处理
        :param max_len: 句子最长的长度，默认为保留100
        :param data_type: 数据类型，当前支持四种数据类型
        """

        self.w2i = get_w2i()  # word to index
        self.vocab_size = len(self.w2i)
        self.unk_flag = unk_flag
        self.pad_flag = pad_flag
        self.max_len = max_len
        self.model = model
        self.tag_size = 2
        self.vocab_path = path_vocab

        # token_dict = {}
        # with codecs.open(self.dict_path, 'r', 'utf8') as reader:
        #     for line in reader:
        #         token = line.strip()
        #         token_dict[token] = len(token_dict)
        #
        # self.tokenizer = Tokenizer(token_dict)

        self.tokenizer = Tokenizer(self.vocab_path, do_lower_case=True)

        self.base_dir = r'F:\mylearning\fengkong/ner_demo/data/data'

    def get_data(self, one_hot: bool = True) -> ([], [], [], []):
        """
        获取数据，包括训练、测试数据中的数据和标签
        :param one_hot:
        :return:
        """

        # 拼接地址

        path_train = os.path.join(self.base_dir, "train_merge_30.csv")
        path_test = os.path.join(self.base_dir, "train_merge_30.csv")

        # 、读取数据
        if self.model == 'bert':
            train_data, train_label = self.__bert_text_to_index(path_train)
            print('开始处理bert dev数据')
            test_data, test_label = self.__bert_text_to_index(path_test, data_type='dev')
        else:
            # train_data, train_label = self.__tokenizer_text_to_indexs(path_train, data_type="dev")
            # print("开始处理dev数据 ")
            # test_data, test_label = self.__tokenizer_text_to_indexs(path_test, data_type="dev")

            train_data, train_label = self.__word_punct_tokenizer_text_to_indexs(path_train)
            print("开始处理dev数据 ")
            test_data, test_label = self.__word_punct_tokenizer_text_to_indexs(path_test, data_type="dev")

        # 进行 one-hot处理
        if one_hot:
            def label_to_one_hot(index: []) -> []:
                print(index)
                data = []
                for line in index:
                    line_line = [0] * 2
                    line_line[line] = 1
                    data.append(line_line)
                return np.array(data)

            train_label = label_to_one_hot(index=train_label)
            test_label = label_to_one_hot(index=test_label)
        else:
            train_label = np.expand_dims(train_label, 1)
            test_label = np.expand_dims(test_label, 1)
        return train_data, train_label, test_data, test_label

    def num2tag(self):
        return dict(enumerate(self.tag))

    def i2w(self):
        return dict(zip(self.w2i.values(), self.w2i.keys()))

    # texts 转化为 index序列
    def __text_to_indexs(self, file_path: str, data_type="train") -> ([], []):
        df_content = pd.read_csv(file_path, sep="\t")
        df_content.dropna(subset=["f_sms_data"], inplace=True)
        df_train = df_content["f_sms_data"].tolist()

        df_train = [re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                           item.lower())
                    for item in df_train]

        datas = []
        labels = []
        for index, item in df_content.iterrows():

            line_data = []
            sms_data = item["f_sms_data"]
            label_one = item["d7"]

            sms_data = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                              sms_data.lower())

            word_tokens = word_tokenize(sms_data)

            for item_ in word_tokens:

                if re.search(r'(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d', item_):
                    result = result[0].split(":")
                    for item__ in result:
                        char_index = self.w2i.get(item__, self.w2i[self.unk_flag])
                        line_data.append(char_index)
                elif len(item_.split("-")) >= 2:
                    result = item_.split("-")
                    for item__ in result:
                        char_index = self.w2i.get(item__, self.w2i[self.unk_flag])
                        line_data.append(char_index)

                elif len(item_.split("/")) >= 2:
                    result = item_.split("/")
                    for item__ in result:
                        char_index = self.w2i.get(item__, self.w2i[self.unk_flag])
                        line_data.append(char_index)
                else:
                    char_index = self.w2i.get(item_, self.w2i[self.unk_flag])
                    line_data.append(char_index)

            if len(line_data) < self.max_len:
                pad_num = self.max_len - len(line_data)
                line_data = line_data + [self.pad_index] * pad_num
            else:
                line_data = line_data[:self.max_len]
            datas.append(line_data)
            labels.append(label_one)

        # np.array(xs[idx:])
        return np.array(datas), np.array(labels)

    def __tokenizer_text_to_indexs(self, file_path: str, data_type="train") -> ([], []):
        df_content = pd.read_csv(file_path, sep="\t")
        df_content.dropna(subset=["f_sms_data"], inplace=True)
        print(f"{data_type} 数据长度：{len(df_content)}")
        datas = []
        labels = []
        for index, item in df_content.iterrows():
            sms_data = item["f_sms_data"]
            label_one = item["d7"]

            ## 去除邮箱
            # sms_data = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
            #                   sms_data.lower())

            line_token_ids = self.tokenizer.tokens_to_ids(sms_data)[1:-1]
            print(len(line_token_ids))

            # print(f"标签 ：{sms_data} \n {self.tokenizer.tokenize(sms_data)}")
            # print(f" ========== \n")

            if len(line_token_ids) >= self.max_len:  # 先进行截断
                line_token_ids = line_token_ids[:self.max_len]

            # padding
            else:  # 填充到最大长度
                pad_num = self.max_len - len(line_token_ids)
                line_token_ids = line_token_ids + [self.w2i.get("[PAD]")] * pad_num

            datas.append(line_token_ids)

            labels.append(label_one)

        # np.array(xs[idx:])
        return np.array(datas), np.array(labels)

    def __word_punct_tokenizer_text_to_indexs(self, file_path: str, data_type="train") -> ([], []):
        df_content = pd.read_csv(file_path, sep="\t")
        df_content.dropna(subset=["f_sms_data"], inplace=True)
        print(f"{data_type} 数据长度：{len(df_content)}")
        datas = []
        labels = []
        tk = WordPunctTokenizer()

        for index, item in df_content.iterrows():
            sms_data = item["f_sms_data"]
            label_one = item["d7"]
            line_token_ids = []
            for line in sms_data.split("\n")[:-1]:
                line = line.split(" , el tiempo ")

                # 去除网址
                txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                             line[0])

                # 去除短信内容的数据 保留日期和具体时间等数字
                txt = re.sub('[0-9]', ' ', txt)

                if len(line) >= 2:
                    line = txt + " , el tiempo " + line[1]
                else:
                    line = txt

                tk_lines = tk.tokenize(line)

                for tk_line in tk_lines:
                    line_token_ids.append(self.w2i.get(tk_line, self.w2i.get("[UNK]")))
            print(f"该用户的token 数 ：{len(line_token_ids)}")
            if len(line_token_ids) >= self.max_len:  # 先进行截断
                line_token_ids = line_token_ids[:self.max_len]

            # padding
            else:  # 填充到最大长度
                pad_num = self.max_len - len(line_token_ids)
                line_token_ids = line_token_ids + [self.w2i.get("[PAD]")] * pad_num

            datas.append(line_token_ids)
            labels.append(label_one)

        # np.array(xs[idx:])
        return np.array(datas), np.array(labels)

    def __bert_text_to_index(self, file_path: str, data_type="train"):
        """
        :param file_path:  文件路径
        :return: [ids, types], label_ids
        """
        data_ids = []
        data_types = []
        label_ids = []
        df = pd.read_csv(file_path)[:20]

        for index, row in df.iterrows():
            tag_index = row[2]
            row = str(row[1])
            # bert 需要输入index和types
            token_ids, seg_ids = self.tokenizer.encode(first=row, max_len=self.max_len)

            # # 处理填充开始和结尾 bert 输入语句每个开始需要填充[CLS] 结束[SEP]
            # if len(token_ids) >= max_len_buff:  # 先进行截断
            #     token_ids = token_ids[:max_len_buff]
            #     token_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
            #
            # # padding
            # else:  # 填充到最大长度
            #     pad_num = max_len_buff - len(token_ids)
            #     token_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id] * pad_num

            assert len(token_ids) == len(seg_ids)
            data_ids.append(token_ids)
            data_types.append(seg_ids)
            label_ids.append(tag_index)
        return [np.array(data_ids), np.array(data_types)], np.array(label_ids)


if __name__ == '__main__':
    import pandas as pd

    dp = DataProcess( max_len=12800)
    x_train, y_train, x_test, y_test = dp.get_data(one_hot=False)
    # print(x_train[1].shape)
    # print(x_train[0].shape)
    #
    # print(y_train.shape)
    # print(x_test[0].shape)
    # print(y_test.shape)
