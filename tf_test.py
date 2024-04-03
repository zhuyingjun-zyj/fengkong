from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf, json

sentences = ["Hello, nice to meet you. \n",
             "Nice to meet you too!", "Hello, Bob"]

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=1,
    filters=[],
    lower=True, split=' ', char_level=False, oov_token=None,
)
tokenizer.fit_on_texts(sentences)

print("tokenizer config:\n", tokenizer.get_config())
print('texts_to_sequences:\n', tokenizer.texts_to_sequences(['Bob, nice to meet you. \n']))

print(tokenizer.get_config(), type(tokenizer.get_config()))
with open("./tokenizer_sms.json", "w", encoding="utf-8") as wf:
    json.dump(tokenizer.get_config(), wf, indent=2,ensure_ascii=False)

# with open("./tokenizer_sms.json", "r", encoding="utf-8") as wf:
#     print(json.load(wf))

