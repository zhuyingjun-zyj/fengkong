import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # ��ȡ��ǰ��ַ
# current_dir = r'E:\project\test\train_model\ner_demo\data\vocab'
# �ʱ�Ŀ¼(���ʱ���õ��� bertԤѵ��ģ�͵Ĵʱ�)
# path_vocab = os.path.join(current_dir, '../data/vocab/vocab.txt')

# path_vocab = os.path.join(current_dir, '../data/vocab/english_char_to_id.json')imdb_word_index.json
path_vocab = os.path.join(current_dir, '../data/vocab/imdb_word_index.json')


# ʵ������ʶ���ļ�Ŀ¼
path_data_dir = os.path.join(current_dir, '../data/data')

path_data2_dir = os.path.join(current_dir, '../data/data2/')
path_msra_dir = os.path.join(current_dir, '../data/MSRA/')
path_renmin_dir = os.path.join(current_dir, '../data/renMinRiBao/')

# bert Ԥѵ���ļ���ַ
path_bert_dir = os.path.join(current_dir, '../data/uncased_L-12_H-768_A-12/')

# ��־����¼���ļ�Ŀ¼��ַ
path_log_dir = os.path.join(current_dir, "../log")

# ģ�ͱ���·��
# path_model_idcnn = os.path.join(current_dir,"../save_model/idcnn/")
# path_model_idcnn = os.path.join(current_dir,"../save_model/idcnn/sparse/")
path_model = os.path.join(current_dir,"../save_model/")


