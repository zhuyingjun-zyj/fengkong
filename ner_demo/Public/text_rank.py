# // 以下代码基于Python3.7，需要的库均为pip安装，部分库安装需要科学上网。亲测无bug，可以直接运行。
# // 注释偏好为写在相关代码下方

import networkx
# 一个图结构的相关操作包，没用过无所谓，有兴趣可以搜索学习
import numpy as np
import pandas as pd

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('punkt')
# nltk.download('stopwords')

# 下载断句和停用词数据，下载一次就行，后续运行可直接注释掉
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# 获取词向量
# 该词向量文件形式为：词 空格 词向量，然后换行，自行理解上述操作代码
word_embeddings = {}
GLOVE_DIR = '/Users/zhuyingjun/Desktop/glove.6B/glove.6B.100d.txt'
with open(GLOVE_DIR, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs


def get_rank(x):
    # print("传进来的数据类型：", type(x), "传进来的数据为：", x)
    x_ = x.split("\n\n")
    result = []
    for txt1 in x_:
        txt = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', txt1)
        txt = re.sub(r"([:?()''<>/*#-]){1,}", r" ", txt)
        # txt = re.sub('[0-9]', '', txt)  # 去数字替换为x
        txt = txt.lower()  # 统一小写
        result.append(txt)
    sentences = list(set(result))
    sentences.sort(key=result.index)
    messages_token= []
    for item  in  sentences:
        sentence = sent_tokenize(item)
        messages_token.extend(sentence)

    print("分句后的：",messages_token)
    print("分句后长度：",len(messages_token))
    # print("+++++++++++++++++++++++")
    # stop_words = stopwords.words('english')
    # def remove_stopwords(str):
    #     sen = ' '.join([i for i in str if i not in stop_words])
    #     return sen
    # clean_sentences = [remove_stopwords(r.split()) for r in sentences]

    clean_sentences = messages_token
    # 去停用词
    sentences_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum(
                [word_embeddings.get(w, np.zeros((100,))) for w in i.split()]
            ) / (len(i.split()) + 1e-2)
        else:
            v = np.zeros((100,))
        sentences_vectors.append(v)

    # 获取每个句子的所有组成词的向量（从GloVe词向量文件中获取，每个向量大小为100），
    # 然后取这些向量的平均值，得出这个句子的合并向量为这个句子的特征向量

    similarity_matrix = np.zeros((len(clean_sentences), len(clean_sentences)))

    # 初始化相似度矩阵（全零矩阵）
    for i in range(len(clean_sentences)):
        for j in range(len(clean_sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(
                    sentences_vectors[i].reshape(1, -1), sentences_vectors[j].reshape(1, -1)
                )

    # 计算相似度矩阵，基于余弦相似度
    nx_graph = networkx.from_numpy_array(similarity_matrix)
    scores = networkx.pagerank_numpy(nx_graph)

    # 将相似度矩阵转为图结构
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True
    )
    result = []
    # 排序
    ranked_sentences = ranked_sentences[:15]

    for i in range(len(ranked_sentences)):
        datas = ranked_sentences[i][1].replace("\n"," ")
        result.append(datas)
    result = ". ".join(result)
    print("ranked_sentences  len ", len(ranked_sentences),"result len ",len(result.split(" ")))
    return result

df = pd.read_csv(r"/Users/zhuyingjun/Desktop/fengkong/ner_demo/DataProcess/1/concat_meger_data_we_.csv")
# data_path = r"/Users/zhuyingjun/Desktop/content_test.csv"
# df = pd.read_csv(data_path)

df["rank_data"] = df['concat'].apply(lambda x: get_rank(x))
df.to_csv(r"/Users/zhuyingjun/Desktop/rank_result.csv",index = False)
# 打印得分最高的前10个句子，即为摘要
