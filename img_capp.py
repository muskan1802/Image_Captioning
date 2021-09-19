import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.preprocessing.sequence import  pad_sequences
from nltk.lm import vocabulary
from tensorflow.keras.utils import to_categorical
from keras.layers import Input,Dense,Dropout,Embedding,LSTM
from keras.layers.merge import  add


def readTExtFile(path):
    with open(path) as f:
        captions = f.read()
    return captions
captions = readTExtFile("archive/Flickr_Data/Flickr_TextData/Flickr8k.token.txt")
# print(len(captions.split("\n")))
captions = captions.split('\n')[:-1]
# print(len(captions))

#dictionary to map each image with the list of captions it has

description = {}

for x in captions:
    first,second = x.split('\t')
    img_name = first.split(".")[0]

    #if image id is already present or not
    if description.get(img_name) is None:
        description[img_name] = []

    description[img_name].append(second)

# print(description["1000268201_693b08cb0e"])
IMG_PATH = "archive/Flickr_Data/Images/"

img = cv2.imread(IMG_PATH + "2089122314_40d5739aef.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
# plt.show()

#DATA CLEANING
# all lower case, remove numbers, remove punctuations , reduce cab size , less overfitting less computation

def clean_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[^a-z]+"," ",sentence)
    sentence = sentence.split()

    sentence = [s for s in sentence if len(s)>1]
    sentence = " ".join(sentence)
    return sentence

# print(clean_text("A cat is sitting over the house # 64"))

# clean all captions
for key,caption_list in description.items():
    for i in range(len(caption_list)):
        caption_list[i] = clean_text(caption_list[i]) # iterating over images and clean the caption

# print(description["1000268201_693b08cb0e"])

#write the data to text file

with open("description.txt","w") as f:
    f.write(str(description))

#vocabulary
#vocab - set of all unique words your model can predict because we going to  make a mapping of word

description = None
with open("description.txt",'r') as f:
    description = f.read()

# Java Script Object Notation (JSON) is a light weight data format with many similarities to python dictionaries.
# JSON objects are useful because browsers can quickly parse them, which is ideal for transporting data between a client and a server.

json_acceptable_string = description.replace("'","\"")
description = json.loads(json_acceptable_string)
print(type(description))

#vocab

vocab = set()
for key in description.keys():
    [vocab.update(sentence.split()) for sentence in description[key]]

# print("Vocab size : %d"% len(vocab))

#total words across all the sentences

total_words = []
for key in description.keys():
    [total_words.append(i) for des in description[key] for i in des.split()]

# print("Total words %d"%len(total_words))

#shorten program - filter words from the vocab according to the certain threshold frequency

import collections

counter = collections.Counter(total_words)
freq_cnt = dict(counter)
# print(freq_cnt)

#short this dictionary according to the freq count

sorted_freq_cnt = sorted(freq_cnt.items(),reverse=True,key=lambda x:x[1])

#filter
threshold = 10
sorted_freq_cnt = [x for x in sorted_freq_cnt if x[1]>threshold]
total_words = [x[0] for x in sorted_freq_cnt]

# print(len(total_words))  #removed duplicates(unique words) , sorted using frequency

#Prepare train/test model

train_file_data = readTExtFile("archive/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt")
test_file_data = readTExtFile("archive/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt")

train = [row.split(".")[0] for row  in train_file_data.split("\n")[:-1]]
test = [row.split(".")[0] for row  in test_file_data.split("\n")[:-1]]

#prepare description for the training data
# tweak - add start (<s>) and end (<e>) token to our training data

train_description = {}
for img_id in train :
    train_description[img_id] = []
    for cap in description[img_id]:
        cap_to_append = "startseq " + cap + " endseq"
        train_description[img_id].append(cap_to_append)

# print(train_description["1000268201_693b08cb0e"])

#transfer learning : image->features , text->features
# step-1 : image feature extraction(resnet50 is pretrained model have 50 layers , has skip connection , gradients can flow and can back propogate easily

model = ResNet50(weights="imagenet",input_shape=(224,224,3))
# print(model.summary())
#instead of taking entire model we'll be using convolutional base
model_new = Model(model.input,model.layers[-2].output)
#preprocess some images where resnet50 expect it to be and then we'll store those image to compute image-feature

def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0) # when we load img it is of 224,224,3 when we feed image in model eg resnet50 we can't feed single img it should be of some batch - 4D tensor(3D to 4D)
    #Normalisation
    img = preprocess_input(img)
    return img

# img = preprocess_img(IMG_PATH + "1000268201_693b08cb0e.jpg")
# plt.imshow(img[0])
# plt.show()

def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    # print(feature_vector.shape)
    return feature_vector

# print(encode_image(IMG_PATH + "1000268201_693b08cb0e.jpg"))

start = time()
encoding_train = {}
#image_id -> feature_vector extracted from resnet image

for ix,img_id in enumerate(train):
    img_path = IMG_PATH+"/"+img_id+".jpg"
    encoding_train[img_id] = encode_image(img_path)

    if ix%100 == 0:
        print("Encoding in Progress Time step %d "%ix)

end_t = time()
print("Total time taken :", end_t-start)

#stroe everything to the disk - pickle is used for that allows us to convert (store ram data to disk)

with open("encoded_train_features.pkl","wb") as f:
    pickle.dump(encoding_train,f)

start_t = time()
encoding_test = {}
#image_id -> feature_vector extracted from resnet image

for ix,img_id in enumerate(test):
    img_path = IMG_PATH+"/"+img_id+".jpg"
    encoding_test[img_id] = encode_image(img_path)

    if ix%100 == 0:
        print("Test Encoding in Progress Time step %d "%ix)

end_tt = time()
print("Total time taken(test) :", end_tt-start_t)

#store everything to the disk - pickle is used for that allows us to convert (store ram data to disk)

with open("encoded_test_features.pkl","wb") as f:
    pickle.dump(encoding_test,f)

# DATA PER-PROCESSING FOR CAPTIONS - since every word is a feature and it needs to be represented using no

#Vocab

word_to_idx = {}
idx_to_word = {}

for i,word in enumerate(total_words):
    word_to_idx[word] = i
    idx_to_word[i+1] = word

#Two special words

idx_to_word[1846] = 'startseq'
word_to_idx['startseq'] = 1846

idx_to_word[1847] = 'endseq'
word_to_idx['endseq'] = 1847

vocab_size = len(word_to_idx)+1
# print("vocab_size",vocab_size)
max_len=0
for key in train_description.keys():
    for cap in train_description[key]:
        max_len = max(max_len,len(cap.split()))

## Data Loader(GENERATOR)
#language modeling - P(wt+1|w1....wt)


def data_generator(train_descriptions, encoding_train, word_to_idx, max_len, batch_size):
    X1, X2, y = [], [], []

    n = 0

    while True:

        for key, desc_list in train_descriptions.items():
            n += 1

            photo = encoding_train[key]

            for desc in desc_list:

                seq = [word_to_idx[word] for word in desc.split() if word in word_to_idx]

                for i in range(1, len(seq)):
                    in_seq = seq[0:i]
                    out_seq = seq[i]

                    in_seq = pad_sequences([in_seq], maxlen=max_len, value=0, padding='post')[0]

                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            if n == batch_size:
                yield {'input_img': np.array(X1), 'input_cap': np.array(X2)}, np.array(y)
                X1, X2, y = [], [], []
                n = 0

f=open("glove.6B.50d.txt",encoding='utf-8')#For windows,we will specify encoding='utf-8'
for line in f:
    values=line.split()
    print(values)
    break
import numpy as np

embedding_index={}#It will store word-vector for every word.
for line in f:
    values=line.split()
    word=values[0]
    word_embedding=np.array(values[1:],dtype='float')
    embedding_index[word]=word_embedding

    #Whenever we pass data to RNN/LSTM layer,that data must pass through embedding layer.
    #Either we can train as we go or we can preinitialize this layer like using Glove6B50D.txt,
    #But in our work,we don't need these 6 Billion words.
    #So ,We will make a matrix of (vocab_size,50).
    #For each word in vocab,we will have 50 dimensional vector.
    #How to construct this matrix from already trained GloveVectors.
f.close()

def get_embedding_matrix():
    embedding_dimension=50
    Matrix=np.zeros((vocab_size,embedding_dimension))
    for word,index in word_to_idx.items():
        embedding_vector=embedding_index.get(word)
        if embedding_vector is not None:
            Matrix[index]=embedding_vector
    return Matrix

embedding_matrix=get_embedding_matrix()
print(type(embedding_matrix))

from tensorflow.keras import layers


#for images
input_img_features = Input(shape=(2048,),name="input_img")
inp_img1 = Dropout(0.3)(input_img_features)
inp_img2 = Dense(256,activation='relu')(inp_img1)

# captions as input

input_captions = Input(shape=(max_len,),name="input_cap")
inp_cap1 = Embedding(input_dim=vocab_size,output_dim=50,mask_zero=True)(input_captions)
inp_cap2 = Dropout(0.5)(inp_cap1)
inp_cap3 = LSTM(256)(inp_cap2)

decoder1 = add([inp_img2,inp_cap3])
decoder2 = Dense(256,activation='relu')(decoder1)
outputs = Dense(vocab_size,activation='softmax')(decoder2)

#combine model

model = Model(inputs=[input_img_features,input_captions],outputs=outputs)

model.summary()



#Important thing

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy',optimizer="adam")
epochs = 10
batch_size = 3
steps = len(train_description)//batch_size
def train_model():
    for i in range(epochs):
        generator=data_generator(train_description,encoding_train,word_to_idx,max_len,batch_size)
        model.fit_generator(generator,epochs=1,steps_per_epoch=steps,verbose=1)
        model.save('./model_weights/model_'+str(i)+'.h5')

train_model()

model = load_model("model_weights/model_9.h5")


# PREDICTION

# def predict_caption(photo):

#   in_text = "startseq"
#   for i in range(max_len):
#     sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
#     sequence = pad_sequences([sequence],maxlen=max_len,padding='post')

#     ypred = model.predict([photo,sequence])
#     ypred = ypred.argmax() #word with max prob always - greedy sampling
#     word = idx_to_word[ypred]
#     in_text += ' ' + word
#     if word == 'endseq':
#       break

#     # final_caption = in_text.split()[1:-1]
#     # final_caption = ' '.join(final_caption)
#     final_caption =  in_text.split()
#     final_caption = final_caption[1:-1]
#     final_caption = ' '.join(final_caption)

#     return final_caption

def predict_caption(photo):
    in_text = "startseq"

    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text += (' ' + word)

        if word == 'endseq':
            break

    final_caption = in_text.split()[1:-1]
    # final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption


# pick some random images and see result

for i in range(2):
    rn = np.random.randint(0, 1000)
    img_name = list(encoding_test.keys())[rn]
    photo = encoding_test[img_name].reshape((1, 2048))

    i = plt.imread("Image_Captioning/archive/Flickr_Data/Images/" + img_name + ".jpg")
    plt.imshow(i)
    plt.axis("off")

    plt.show()
    caption = predict_caption(photo)
    print(caption)
    # plt.imshow(i)
    # plt.axis("off")
    # plt.show()

    # caption = predict_caption(photo)
    # print(caption)
# for i in range(2):
#   idx = np.random.randint(0,1000)
#   all_img_names = list(encoding_test.keys())
#   img_name = list(encoding_test.keys())[idx]
#   photo_2048 = encoding_test[img_name].reshape((1,2048))

#   i = plt.imread("Image_Captioning/archive/Flickr_Data/Images/"+img_name+".jpg")
#   plt.imshow(i)
#   plt.axis('off')
#   plt.show()
#   caption = predict_caption(photo_2048)
#   print(caption)