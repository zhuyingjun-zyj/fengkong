import nltk
from nltk.tokenize import TweetTokenizer,WhitespaceTokenizer,SExprTokenizer,WordPunctTokenizer
import pandas as pd
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize.mwe import MWETokenizer
test_merge_file = r"../data/data/train_merge_30.csv"
# Create a reference variable for Class TweetTokenizer
tk = WordPunctTokenizer()   # F:\mylearning\fengkong\stanford-postagger-full-2017-06-09\stanford-postagger-3.8.0.jar
# tk = StanfordTokenizer(path_to_jar  = r"F:\mylearning\fengkong\stanford-postagger-full-2017-06-09\stanford-postagger-3.8.0.jar")
# spanish_tokenizer = nltk.data.load("tokenizers/punkt/spanish.pickle")

df = pd.read_csv(test_merge_file, sep="\t")
df.dropna(subset=["f_sms_data"], inplace=True)
df = df["f_sms_data"].tolist()

"""
TweetTokenizer : '2022-11-', '23', '18:17'
WhitespaceTokenizer : '2022-11-22', '15:14:26'
StanfordTokenizer : '2022-11-22', '15:14:26'
MWETokenizer   : 'E', 's', 't', 'i', 'm', 'a', 'd', 'o', ' ', 
WordPunctTokenizer : '2022', '-', '11', '-', '19', '08', ':', '23', ':', '11'

"""

for item in df:
    # print(f"原始数据：{item} type :{type(item)}")
    if item != "nan":
        print(item.split("\n"))
        # geek = tk.tokenize(item)  #  '2022-11-', '23', '18:17'
        # print(f"切词长度： {len(geek)} ")
        print("================\n\n")
