import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Input, concatenate, Attention
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import datetime
from nltk.stem import PorterStemmer
import multiprocessing
import json

lemmatizer = WordNetLemmatizer()

# 去除标点
signos_puntuacion = string.punctuation

# 英语停用词
english_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll",
                     "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's",
                     "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs",
                     "themselves", "what", "which", "who", "whom", "this", "that", "that'll", "these", "those", "am",
                     "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
                     "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                     "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during",
                     "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                     "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
                     "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
                     "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't",
                     "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't",
                     "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
                     "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn",
                     "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
                     "won't", "wouldn", "wouldn't"]

# 西班牙语停用词
spanish_stopwords = ["de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con",
                     "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí",
                     "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay",
                     "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
                     "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo",
                     "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos",
                     "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú", "te",
                     "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os", "mío", "mía", "míos", "mías",
                     "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra",
                     "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy",
                     "estás", "está", "estamos", "estáis", "están", "esté", "estés", "estemos", "estéis", "estén",
                     "estaré", "estarás", "estará", "estaremos", "estaréis", "estarán", "estaría", "estarías",
                     "estaríamos", "estaríais", "estarían", "estaba", "estabas", "estábamos", "estabais", "estaban",
                     "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron", "estuviera",
                     "estuvieras", "estuviéramos", "estuvierais", "estuvieran", "estuviese", "estuvieses",
                     "estuviésemos", "estuvieseis", "estuviesen", "estando", "estado", "estada", "estados", "estadas",
                     "estad", "he", "has", "ha", "hemos", "habéis", "han", "haya", "hayas", "hayamos", "hayáis",
                     "hayan", "habré", "habrás", "habrá", "habremos", "habréis", "habrán", "habría", "habrías",
                     "habríamos", "habríais", "habrían", "había", "habías", "habíamos", "habíais", "habían", "hube",
                     "hubiste", "hubo", "hubimos", "hubisteis", "hubieron", "hubiera", "hubieras", "hubiéramos",
                     "hubierais", "hubieran", "hubiese", "hubieses", "hubiésemos", "hubieseis", "hubiesen", "habiendo",
                     "habido", "habida", "habidos", "habidas", "soy", "eres", "es", "somos", "sois", "son", "sea",
                     "seas", "seamos", "seáis", "sean", "seré", "serás", "será", "seremos", "seréis", "serán", "sería",
                     "serías", "seríamos", "seríais", "serían", "era", "eras", "éramos", "erais", "eran", "fui",
                     "fuiste", "fue", "fuimos", "fuisteis", "fueron", "fuera", "fueras", "fuéramos", "fuerais",
                     "fueran", "fuese", "fueses", "fuésemos", "fueseis", "fuesen", "sintiendo", "sentido", "sentida",
                     "sentidos", "sentidas", "siente", "sentid", "tengo", "tienes", "tiene", "tenemos", "tenéis",
                     "tienen", "tenga", "tengas", "tengamos", "tengáis", "tengan", "tendré", "tendrás", "tendrá",
                     "tendremos", "tendréis", "tendrán", "tendría", "tendrías", "tendríamos", "tendríais", "tendrían",
                     "tenía", "tenías", "teníamos", "teníais", "tenían", "tuve", "tuviste", "tuvo", "tuvimos",
                     "tuvisteis", "tuvieron", "tuviera", "tuvieras", "tuviéramos", "tuvierais", "tuvieran", "tuviese",
                     "tuvieses", "tuviésemos", "tuvieseis", "tuviesen", "teniendo", "tenido", "tenida", "tenidos",
                     "tenidas", "tened"]


def sort_sms(sms_list):
    try:
        sorted_sms = sorted(sms_list, key=lambda sms: pd.to_datetime(sms['smsTime']))
    except ValueError:
        # 处理日期解析失败的情况，删除包含无效日期的短信，并继续执行排序
        valid_sms = [sms for sms in sms_list if 'smsTime' in sms and is_valid_date(sms['smsTime'])]
        sorted_sms = sorted(valid_sms, key=lambda sms: pd.to_datetime(sms['smsTime']))

    return sorted_sms


def is_valid_date(date_string):
    try:
        pd.to_datetime(date_string)
        return True
    except ValueError:
        return False


def process_sms_txt(sms_data, submit_time):
    submit_time = datetime.datetime.strptime(submit_time, '%Y-%m-%d %H:%M:%S')

    sms_list = json.loads(sms_data) if pd.notnull(sms_data) else []

    sms_list = sort_sms(sms_list)

    sms_list = ' ||| '.join([sms['content'] for sms in sms_list if sms.get('type') == '1' and 'content' in sms and (
                submit_time - pd.to_datetime(sms['smsTime'])).days <= 15])

    return sms_list


def preprocess_text(sms_data, submit_time, languages=["english", "spanish"]):
    txt = process_sms_txt(sms_data, submit_time)

    # 去除网址
    txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', txt)

    # 数字处理
    txt = re.sub('[0-9]', ' ', txt)

    # 统一小写
    txt = txt.lower()
    txt = re.sub('\n', ' ', txt)

    # 创建一个翻译表，将标点符号映射为空格
    tabla_de_traduccion = txt.maketrans(signos_puntuacion, ' ' * len(signos_puntuacion))

    # 使用translate方法将标点符号替换为空格
    txt = txt.translate(tabla_de_traduccion)

    # 分词
    words = word_tokenize(txt)

    # 遍历词汇并进行词形还原
    reconstructed_words = []
    for word in words:
        # 使用词形还原器将词汇还原为基本形式（默认为英语）
        lemma = lemmatizer.lemmatize(word)
        reconstructed_words.append(lemma)

    # 去除停用词
    stop_words = set()
    for language in languages:
        if language == "english":
            stop_words.update(english_stopwords)
        elif language == "spanish":
            stop_words.update(set(spanish_stopwords))

    filtered_words = [word for word in reconstructed_words if word not in stop_words]

    return ' '.join(filtered_words)


def read_csv(file):
    df = pd.read_csv(file, sep='/001XG')
    for item in df["f_sms_data"]:
        print(item)
        print("\n ==============  \n")
    # df.to_csv("./test.csv", index=False)


if __name__ == '__main__':
    # read_csv(r"F:\mylearning\fengkong\数据\data_sma_old_test.csv")
    # import datetime
    #
    # smsTime=  "2023-09-17 23:09:47"
    # data_format = "%Y-%m-%d %H:%M:%S"
    # DATA = datetime.datetime.strptime(smsTime,data_format)
    #
    # import calendar
    #
    # date_str = '2024-04-01'
    # year, month, day = map(int, date_str.split('-'))
    # weekday = calendar.weekday(year, month, day)
    # weekday_str = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'][weekday]
    # print(f'{date_str}是{weekday_str}')

    print(re.sub(r'^(?:[01]\d|2[0-3]): [0 - 5]\d:[0 - 5]\d$', '', "'10:10:08': 24974"))


