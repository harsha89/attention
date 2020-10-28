from tensorflow.keras.layers import Input, Activation, Flatten, Dense, LSTM, Lambda,Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import json
import pandas as pd
import os

class_labels = {
    "drugs-narcotics": 0,
    "extremism": 1,
    "finance": 2,
    "cash-in": 3,
    "cash-out": 4,
    "hacking": 5,
    "identification-credentials": 6,
    "intellectual-property-copyright-materials": 7,
    "pornography-adult": 8,
    "pornography-child-exploitation": 9,
    "pornography-illicit-or-illegal": 10,
    "search-engine-index": 11,
    "unclear": 12,
    "violence": 13,
    "weapons": 14,
    "credit-card": 15,
    "counteir-feit-materials": 16,
    "gambling": 17,
    "library": 18,
    "other-not-illegal": 19,
    "legitimate": 20,
    "chat": 21,
    "mixer": 22,
    "mystery-box": 23,
    "anonymizer": 24,
    "vpn-provider": 25,
    "email-provider": 26,
    "escrow": 27,
    "softwares": 28,
    "education-training": 29,
    "file-sharing": 30,
    "forum": 31,
    "wiki": 32,
    "hosting": 33,
    "general": 34,
    "information-sharing-reportage": 35,
    "marketplace-for-sale": 36,
    "recruitment-advocacy": 37,
    "system-placeholder": 38,
    "conspirationist": 39,
    "scam": 40,
    "hate-speech": 41,
    "religious": 42,
    "incomplete": 43,
    "captcha": 44,
    "LoginForms": 45,
    "police-notice": 46,
    "test": 47,
    "legal-statement": 48,
    "whistleblower": 49,
    "error_page": 50,
    "other": 51
}

file_path_0 = 'C:\\Users\\20522516\\phd\\RESEARCH\\CYBER_DATASETS\\DATASET_DEEP_WEB\\labels_0.json'
file_path_1 = 'C:\\Users\\20522516\\phd\\RESEARCH\\CYBER_DATASETS\\DATASET_DEEP_WEB\\labels_1.json'
file_path_0_images = 'C:\\Users\\20522516\\phd\\RESEARCH\\CYBER_DATASETS\\DATASET_DEEP_WEB\\folder_0\\'
file_path_1_images = 'C:\\Users\\20522516\\phd\\RESEARCH\\CYBER_DATASETS\\DATASET_DEEP_WEB\\folder_1_SECONDPASS\\'


list_0_image_deep_web = []
list_0_image_loc = []
list_0_image_name = []
list_0_converted = []
list_1_image_deep_web = []
list_1_image_loc = []
list_1_image_name = []
list_1_converted = []

path_converted_1 = "../../azure/folder_image_set_1\\"
path_converted_2 = "../../azure/folder_image_set_2\\"

def get_converted_text(name, path_folder):
    image_name_plain, ext = os.path.splitext(name)
    converted_loc_json = path_folder + image_name_plain + ".json"
    text = " "

    if not os.path.isfile(converted_loc_json):
        print("=========================================")
        print(converted_loc_json)
        print(text)
        print("=========================================")
        return text

    with open(converted_loc_json, encoding="utf8") as json_file:
        data = json.load(json_file)
        print(converted_loc_json)
        result = data['analyzeResult']
        if result != None:
            ocr_results = result['readResults']
            if ocr_results != None:
                for ocr_result in ocr_results:
                    lines = ocr_result['lines']
                    for line in lines:
                        text = text + line['text'] + " "
        print("=========================================")
        print(converted_loc_json)
        print(text)
        print("=========================================")
        return text

with open(file_path_0, encoding="utf8") as json_file:
    data = json.load(json_file)
    for entry in data:
        meta = []
        list_0_image_name.append(entry)
        list_0_image_loc.append(file_path_0_images + entry)
        for meta_data in data[entry]:
            if len(meta_data.split("=")) > 1:
                type_image = meta_data.split("=")[0]
                type_detail = meta_data.split("=")[1]
                type_detail = type_detail.replace('"', '')
                meta.append(type_detail)
            else:
                meta.append(meta_data)
        converted_text = get_converted_text(entry, path_converted_1)
        list_0_converted.append(converted_text)
        list_0_image_deep_web.append(','.join(meta))


with open(file_path_1, encoding="utf8") as json_file:
    data = json.load(json_file)
    for entry in data:
        meta = []
        list_1_image_name.append(entry)
        list_1_image_loc.append(file_path_1_images + entry)
        for meta_data in data[entry]:
            if len(meta_data.split("=")) > 1:
                type_image = meta_data.split("=")[0]
                type_detail = meta_data.split("=")[1]
                type_detail = type_detail.replace('"', '')
                meta.append(type_detail)
            else:
                meta.append(meta_data)
        converted_text = get_converted_text(entry, path_converted_2)
        list_1_converted.append(converted_text)
        list_1_image_deep_web.append(','.join(meta))

combined_name_list = list_0_image_name + list_1_image_name
combined_loc_list = list_0_image_loc + list_1_image_loc
combined_tags_list = list_0_image_deep_web + list_1_image_deep_web
combined_con_list = list_0_converted + list_1_converted

data = {'image_name': combined_name_list,
        'image_location': combined_loc_list,
        'deep_web_tags': combined_tags_list,
        'content': combined_con_list
        }

df = pd.DataFrame(data, columns = ['image_location', 'image_name', 'deep_web_tags', 'content'])
print(df)
df.to_csv("deep_web.csv", index=False)


# vocab_size = 20000  # Only consider the top 20k words
# maxlen = 200  # Only consider the first 200 words of each movie review
# (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
# print(len(x_train), "Training sequences")
# print(len(x_val), "Validation sequences")
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
#
# news_input = Input(shape=(None,),dtype=tf.string)
# news_embedding = layers.Lambda(make_news_embedding, output_shape=(None,128))(news_input)
# news_embedding = layers.Reshape((-1,128))(news_embedding)
# x = LSTM(128)(news_embedding)
#
# img_input = Input(shape=(7,7,512))
# y = Flatten()(img_input)
# z = Concatenate()([x,y])
#
#
# z = Dense(64, activation='relu')(z)
# z = Dense(32, activation='relu')(z)
# predictions = Dense(5, activation='softmax')(z)
#
# model = Model(inputs=[news_input, img_input], outputs=predictions)
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['acc'])
#
# model.summary()