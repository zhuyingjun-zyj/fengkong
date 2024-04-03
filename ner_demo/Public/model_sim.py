import pandas as pd

fast_text_path = r'/Users/zhuyingjun/Desktop/test_sms_fastext_is_same.csv'
df_fast = pd.read_csv(fast_text_path)
print("\n原始数据：\n ", df_fast.shape)
df_fast = df_fast[~(df_fast.sms_fastext_30_150_score == -999)]
print("\n处理完-999后：\n ", df_fast.shape)
df_fast = df_fast[["uid", "sms_fastext_30_150_score"]]
print("\nfast_test的得分数据：\n ", df_fast.shape, df_fast.columns)

data_me_path = "/Users/zhuyingjun/Desktop/fengkong/ner_demo/data/data/1/200/10_11_we_200_token.csv"
data_train = pd.read_csv(data_me_path)
print("\n第二次给的所有短信数据 all_data：\n ", data_train.shape, data_train.columns)

user_list_m_path = "/Users/zhuyingjun/Desktop/短信_1028_1111/user_list_m.csv"
user_list_m = pd.read_csv(user_list_m_path)
print("\n第二次给的所有短信数据 user_list_m：\n ", user_list_m.shape, user_list_m.columns)
token_t_train = data_train.merge(user_list_m[["uid", "device_id"]], left_on='uid', right_on='device_id', how="left")
print("\n合并后：\n ", token_t_train.shape, token_t_train.columns)


token_t_train = token_t_train.merge(df_fast[['uid', 'sms_fastext_30_150_score']], left_on='uid_y', right_on='uid')
print("\n得分后合并：\n ", token_t_train.shape, token_t_train.columns)
token_t_train[['tokens', 'd7', 'repayment_time', 'device_id', 'uid',
               'sms_fastext_30_150_score']].to_csv("./fast_hebing.csv", index=False)
print("\n得分后合并：\n ", token_t_train.shape, token_t_train.columns)
