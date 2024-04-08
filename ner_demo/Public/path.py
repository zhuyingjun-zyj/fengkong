import os
current_dir = os.path.dirname(os.path.abspath(__file__))  #

current_dir = os.path.dirname(os.path.abspath(__file__))  # ��ȡ��ǰ��ַ
# current_dir = r'E:\project\test\train_model\ner_demo\data\vocab'
# �ʱ�Ŀ¼(���ʱ���õ��� bertԤѵ��ģ�͵Ĵʱ�)
# path_vocab = os.path.join(current_dir, '../data/vocab/vocab.txt')

# path_vocab = os.path.join(current_dir, '../data/vocab/english_char_to_id.json')imdb_word_index.json
# path_vocab = os.path.join(current_dir, '../data/vocab/imdb_word_index.json')
path_vocab = os.path.join(current_dir, '../data/vocab/vocab_spanish')


# ʵ������ʶ���ļ�Ŀ¼
path_data_dir = os.path.join(current_dir, '../data/data')

path_data2_dir = os.path.join(current_dir, '../data/data2/')
path_msra_dir = os.path.join(current_dir, '../data/MSRA/')
path_renmin_dir = os.path.join(current_dir, '../data/renMinRiBao/')

#
path_bert_dir = os.path.join(current_dir, '../data/uncased_L-12_H-768_A-12/')

#
path_log_dir = os.path.join(current_dir, "../log")

#

path_save_model = os.path.join(current_dir,"../save_model/WordPunctTokenizer/")
print(path_save_model)


