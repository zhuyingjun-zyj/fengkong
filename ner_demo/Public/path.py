import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # ��ȡ��ǰ��ַ
# current_dir = r'E:\project\test\train_model\ner_demo\data\vocab'
# �ʱ�Ŀ¼(���ʱ���õ��� bertԤѵ��ģ�͵Ĵʱ�)
# path_vocab = os.path.join(current_dir, '../data/vocab/vocab.txt')

# path_vocab = os.path.join(current_dir, '../data/vocab/english_char_to_id.json')imdb_word_index.json
# path_vocab = os.path.join(current_dir, '../data/vocab/imdb_word_index.json')
tokenzier_path_vocab = os.path.join(current_dir, '../data/vocab/vocab_spanish')
tokenzier_path_vocab_clean = os.path.join(current_dir, '../DataProcess/clean')


# ʵ������ʶ���ļ�Ŀ¼
path_data_dir = os.path.join(current_dir, '../data/data')

path_data2_dir = os.path.join(current_dir, '../data/data2/')
path_msra_dir = os.path.join(current_dir, '../data/MSRA/')
path_renmin_dir = os.path.join(current_dir, '../data/renMinRiBao/')



#
log_path = os.path.join(current_dir, '../logs/')

save_model_path = os.path.join(current_dir,"../save_model/")

# wordpunk 的分词器
tokenizer_sms_all_path = os.path.join(current_dir,"../DataProcess/tokenizer_sms_all.json")

model_version_info = "idcnn_nltk_word_punk"

max_len = 12800
batch_size = 128
epochs = 30

load_model_weight = os.path.join(current_dir,
                                 '../model/save_train_model/electra_small_b128_e20_train_mycheck6/24_epoch.weight')

