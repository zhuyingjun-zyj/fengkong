from gensim.models import Word2Vec
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
from gensim import models
from tqdm import tqdm

train_merge_file = r"../data/data/train_clean.csv"
test_merge_file = r"../data/data/test_clean.csv"

tk = WordPunctTokenizer()  # F:\mylearning\fengkong\stanford-postagger-full-2017-06-09\stanford-postagger-3.8.0.jar

# df_train = pd.read_csv(train_merge_file, sep="\t")
# df_train.dropna(subset=["f_sms_data"], inplace=True)
# df_train = df_train["f_sms_data"].tolist()

df_test = pd.read_csv(test_merge_file, sep="\t")
df_test.dropna(subset=["f_sms_data"], inplace=True)

df_test = df_test["f_sms_data"].tolist()

# df_train.extend(df_test)

# import gc
#
# del df_test
# gc.collect()

print(f"共 {len(df_test)} 条短信")
lines = []
for item in tqdm(df_test):
    for line in item.split("\n")[:-1]:
        geek_line = tk.tokenize(line)
        lines.append(geek_line)

# del df_train
# gc.collect()

# 调用Word2Vec训练 参数：size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
model = Word2Vec(lines, vector_size=64, window=3, min_count=3, epochs=7, negative=10, sg=1)

model.wv.save_word2vec_format('./sms_word2vec.bin', binary=True)
# model = models.KeyedVectors.load_word2vec_format('./sms_word2vec.bin', binary=True)
# print(model.shape)

print("tiempo 的词向量：\n", model.wv.get_vector('tiempo'))
print("\n和孔明相关性最高的前20个词语：")
model.wv.most_similar('tiempo', topn=20)  # 与孔明最相关的前20个词语
