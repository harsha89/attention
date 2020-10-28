import re
import string
from IPython.display import display

import matplotlib.pyplot as plt
import matplotlib.style as style
import nltk
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import os
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow_hub as hub
from utils_pkg.utils import *
from sklearn.calibration import calibration_curve
from datetime import datetime
import numpy as np
from tensorflow.keras.utils import plot_model
import matplotlib.cm as cm


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

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
    "other": 51,
    "videos": 52,
    "ddos-services": 53,
    "political-speech": 54,
    "Gambling": 55
}

class_labels_only_strings = {
    "drugs-narcotics": "drugsnarcotics",
    "extremism": "extremism",
    "finance": "finance",
    "cash-in": "cashin",
    "cash-out": "cashout",
    "hacking": "hacking",
    "identification-credentials": "identificationcredentials",
    "intellectual-property-copyright-materials": "intellectualpropertycopyrightmaterials",
    "pornography-adult": "pornographyadult",
    "pornography-child-exploitation": "pornographychildexploitation",
    "pornography-illicit-or-illegal": "pornographyillicitorillegal",
    "search-engine-index": "searchengineindex",
    "unclear": "unclear",
    "violence": "violence",
    "weapons": "weapons",
    "credit-card": "creditcard",
    "counteir-feit-materials": "counteirfeitmaterials",
    "gambling": "gambling",
    "library": "library",
    "other-not-illegal": "othernotillegal",
    "legitimate": "legitimate",
    "chat": "chat",
    "mixer": "mixer",
    "mystery-box": "mysterybox",
    "anonymizer": "anonymizer",
    "vpn-provider": "vpnprovider",
    "email-provider": "emailprovider",
    "escrow": "escrow",
    "softwares": "softwares",
    "education-training": "educationtraining",
    "file-sharing": "filesharing",
    "forum": "forum",
    "wiki": "wiki",
    "hosting": "hosting",
    "general": "general",
    "information-sharing-reportage": "informationsharingreportage",
    "marketplace-for-sale": "marketplaceforsale",
    "recruitment-advocacy": "recruitmentadvocacy",
    "system-placeholder": "systemplaceholder",
    "conspirationist": "conspirationist",
    "scam": "scam",
    "hate-speech": "hatespeech",
    "religious": "religious",
    "incomplete": "incomplete",
    "captcha": "captcha",
    "LoginForms": "LoginForms",
    "police-notice": "policenotice",
    "test": "test",
    "legal-statement": "legalstatement",
    "whistleblower": "whistleblower",
    "error_page": "error_page",
    "other": "other",
    "videos": "videos",
    "ddos-services": "ddosservices",
    "political-speech": "politicalspeech",
    "Gambling": "Gambling"
}


def remove_punctuation(text):
    text_nopunct = "".join(
        [char for char in text if char not in string.punctuation])  # It will discard all punctuations
    return text_nopunct


def tokenize(text):
    # Match one or more characters which are not word character
    tokens = re.split('\W+', text)
    return tokens


def remove_stopwords(tokenized_list):
    # Remove all English Stopwords
    stopword = nltk.corpus.stopwords.words('english')
    text = [word for word in tokenized_list if word not in stopword]
    return text


def lemmatizing(tokenized_text):
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

#=======================================================================================
df_content = pd.read_csv("deep_web_short.csv")
df_content = df_content.dropna(subset=['deep_web_tags'])
# tokenizer = Tokenizer(num_words=number_of_deep_web_taxonomy, split=' ')
# texts = []
# for tag_list in df_content['deep_web_tags']:
#     print(tag_list)
#     splits = tag_list.split(',')
#     filtered_list = []
#     for split in splits:
#         filtered_list.append(class_labels_only_strings[split])
#     texts.append(' '.join(filtered_list))
# 
# print(texts)
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
# word_index = tokenizer.word_index
# print('Found {} unique tokens'.format(len(word_index)))
# tag_data = pad_sequences(sequences, maxlen=maxlen)
# print(tag_data)
# print(word_index)

def clean_site_content(input_str):
    # print(input_str)
    input_str = input_str.lower()
    # Remove digits
    input_str = re.sub(r'\d+', '', input_str)
    input_str = remove_punctuation(input_str)
    tokenized_string = tokenize(input_str)
    lemmatized_tokens = lemmatizing(tokenized_string)
    input_str = " ".join(lemmatized_tokens)
    # print("formatted", input_str)
    # print("length", len(input_str))
    return input_str


df_content['formatted_content'] = df_content['content'].apply(clean_site_content)

# Get label frequencies in descending order
label_freq = df_content['deep_web_tags'].apply(lambda s: str(s).split(',')).explode().value_counts().sort_values(ascending=False)

# Create a list of rare labels
rare = list(label_freq[label_freq < 0].index) #CHANGED
print("We will be ignoring these rare labels:", rare)

df_content['deep_web_tags'] = df_content['deep_web_tags'].apply(lambda s: [l for l in str(s).split(',') if l not in rare])
# df_content = df_content[df_content['deep_web_tags'].map(lambda d: len(d)) > 0]
df_content.to_csv("formatter_new.csv")

print(df_content.head())

X_train, X_val, y_train, y_val = train_test_split(df_content, df_content['deep_web_tags'], test_size=0.2)
print("Number of posters for training: ", len(X_train))
print("Number of posters for validation: ", len(X_val))

# X_train = [os.path.join('./data/movie_poster/images', str(f)+'.jpg') for f in X_train]
# X_val = [os.path.join('./data/movie_poster/images', str(f)+'.jpg') for f in X_val]
print(X_train[:3])

y_train = list(y_train)
y_val = list(y_val)
print(y_train[:3])

# Bar plot
style.use("fivethirtyeight")
plt.figure(figsize=(12,10))
sns.barplot(y=label_freq.index.values, x=label_freq, order=label_freq.index)
plt.title("Label frequency", fontsize=14)
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

nobs = 8 # Maximum number of images to display
ncols = 4 # Number of columns in display
nrows = nobs//ncols # Number of rows in display

style.use("default")
plt.figure(figsize=(12,4*nrows))
for i in range(nrows*ncols):
    print(i)
    ax = plt.subplot(nrows, ncols, i+1)
    plt.imshow(Image.open(X_train['image_location'].iloc[i]))
    plt.title(y_train[i], size=10)
    plt.axis('off')

plt.show()

# Fit the multi-label binarizer on the training set
print("Labels:")
mlb = MultiLabelBinarizer()
mlb.fit(y_train + y_val)

# Loop over all labels and show them
N_LABELS = len(mlb.classes_)
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))


# transform the targets of the training and test sets
y_train_bin = mlb.transform(y_train)
y_val_bin = mlb.transform(y_val)

# Print example of movie posters and their binary targets
for i in range(3):
    print(X_train['image_location'].iloc[i], y_train_bin[i])

IMG_SIZE = 150 # Specify height and width of image to match the input format of the model
CHANNELS = 3 # Keep RGB color channels to match the input format of the model

training_sentences = []
testing_sentences = []

for row in X_train.iterrows():
    content = row[1]['formatted_content']
    training_sentences.append(content)

for row in X_val.iterrows():
    content = row[1]['formatted_content']
    testing_sentences.append(content)
    
    
vocabulary_size = 10000
embedding_dim = 100
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"
number_of_deep_web_taxonomy = 52
num_heads = 2
ff_dim = 32


tokenizer = Tokenizer(num_words=vocabulary_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding="post")

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type, padding="post")

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

def parse_function(filename, embedding, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, embedding, label

def parse_image_file(filename):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized

def parse_function(filename, label):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label

BATCH_SIZE = 64 # Big enough to measure an F1-score
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 256 # Shuffle the training data by a chunck of 1024 observations


def create_dataset(filenames, labels, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """

    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)

    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

train_images = []
validation_images = []

for row in X_train.iterrows():
    train_images.append(parse_image_file(row[1]['image_location']))

for row in X_val.iterrows():
    validation_images.append(parse_image_file(row[1]['image_location']))


print(decode_review(padded[1]))
print("=============================================================")
print(training_sentences[1])
print("=============================================================")
print(sequences[1])
print("=============================================================")
print(X_train.iloc[1])
print("=============================================================")
save_img('test.jpg', train_images[1])
# load the image to confirm it was saved correctly
# img = load_img('test.jpg')
# img.show()
print("=============================================================")

image_input = Input(shape=(150, 150, 3), name='image_input')
vgg16 = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))(image_input)


x = layers.Flatten()(vgg16)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.1)(x)
x_image = layers.Dense(20, activation='relu')(x)

text_input = Input(shape=(max_length,), dtype='int32', name='embed')
embedding_layer = TokenAndPositionEmbedding(max_length, vocabulary_size, embedding_dim)
x_text = embedding_layer(text_input)
transformer_block = TransformerBlock(embedding_dim, num_heads, ff_dim)
x_text = transformer_block(x_text)
x_text = layers.GlobalAveragePooling1D()(x_text)
x_text = layers.Dropout(0.1)(x_text)
x_text = layers.Dense(20, activation="relu")(x_text)
concatenated = layers.concatenate([x_image, x_text], axis=-1)
output = layers.Dense(N_LABELS, activation='sigmoid')(concatenated)
model = Model([image_input, text_input], output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

LR = 1e-5 # Keep it small when transfer learning
EPOCHS = 0

train_ds = tf.data.Dataset.from_tensor_slices(((train_images, padded), y_train_bin)).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices(((validation_images, testing_padded), y_val_bin)).batch(32)

history = model.fit(train_ds,
                    epochs=EPOCHS,
                    batch_size=32, validation_data=val_ds)
# losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history)


# Get all label names
label_names = mlb.classes_
# Performance table with the first model (macro soft-f1 loss)
grid = perf_grid(val_ds, y_val_bin, label_names, model)


def show_prediction(title, movies_df, model):
    # Get movie info
    image_name = df_content.loc[df_content['image_name'] == title]['image_name'].iloc[0]
    image_loc = df_content.loc[df_content['image_name'] == title]['image_location'].iloc[0]
    content = df_content.loc[df_content['image_name'] == title]['formatted_content'].iloc[0]
    print(image_name)
    print(content)
    tags = df_content.loc[df_content['image_name'] == title]['deep_web_tags'].iloc[0]

    # Read and prepare image
    img = parse_image_file(image_loc)

    images = []
    images.append(img)
    #Content to padded
    test_sequence = tokenizer.texts_to_sequences([content])
    test_padded_seq = pad_sequences(test_sequence, maxlen=max_length, truncating=trunc_type, padding="post")

    y_ground_truth = mlb.transform([tags])

    pred_ds = tf.data.Dataset.from_tensor_slices(((images, test_padded_seq), y_ground_truth)).batch(1)
    # Generate prediction
    prediction = (model.predict(pred_ds) > 0.5).astype('int')
    prediction = pd.Series(prediction[0])
    prediction.index = mlb.classes_
    prediction = prediction[prediction == 1].index.values

    # Dispaly image with prediction
    style.use('default')
    plt.figure(figsize=(8, 4))
    plt.imshow(Image.open(image_loc))
    plt.title('\n\n{}\n\nName\n{}\n\nText\n{}\n\nPrediction\n{}\n'.format(image_name, content, tags, list(prediction)), fontsize=9)
    plt.show()


titles = ["abounding-likeable-bizarre-fact.png",
          "faulty-subdued-absorbed-girlfriend.png",
          "abortive-trashy-deserted-sun.png",
          "past-high-gigantic-western.png",
          "earthy-defiant-ahead-queen.png",
          "anxious-level-distinct-senior.png"]

for t in titles:
    show_prediction(t, df_content, model)

layer_output = model.get_layer('vgg16').get_layer('block3_conv1').output
print(layer_output)

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer('vgg16').get_layer(last_conv_layer_name)
    layer_input  = model.get_layer("vgg16").get_layer("input_1").input
    last_conv_layer_model = keras.Model(layer_input, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


title = 'abounding-likeable-bizarre-fact.png'
image_name = df_content.loc[df_content['image_name'] == title]['image_name'].iloc[0]
image_loc = df_content.loc[df_content['image_name'] == title]['image_location'].iloc[0]
content = df_content.loc[df_content['image_name'] == title]['formatted_content'].iloc[0]
tags = df_content.loc[df_content['image_name'] == title]['deep_web_tags'].iloc[0]

# Read and prepare image
img_size = (150, 150)
img = get_img_array(image_loc, img_size)
images = []
images.append(img)
# Content to padded
test_sequence = tokenizer.texts_to_sequences([content])
test_padded_seq = pad_sequences(test_sequence, maxlen=max_length, truncating=trunc_type, padding="post")

y_ground_truth = mlb.transform([tags])

last_conv_layer_name = 'block5_pool'
classifier_layer_names = ['flatten', 'dense']

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(
    img, model, last_conv_layer_name, classifier_layer_names
)

# Display heatmap
plt.matshow(heatmap)
plt.show()


# We load the original image
img = keras.preprocessing.image.load_img(image_loc)
img = keras.preprocessing.image.img_to_array(img)

# We rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)

# We use jet colormap to colorize heatmap
jet = cm.get_cmap("jet")

# We use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# We create an image with RGB colorized heatmap
jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

# Superimpose the heatmap on original image
superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

# Save the superimposed image
save_path = "out_cam.jpg"
superimposed_img.save(save_path)

# Display Grad CAM
import matplotlib.image as mpimg
image = mpimg.imread(save_path)
plt.imshow(image)
plt.show()