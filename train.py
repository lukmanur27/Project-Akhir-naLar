import nltk
nltk.download('all')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pickle 
import json
import random


text_kata = []
clas = []
dataset = []
ignoreWords = ['!', '?', ',', '.', '-', 'apa', 'sih', 'apasih', 'gimana' ,'apakah', 'saja', 'aja', 'darimana', 'dari', 'mana', 'di', 'bagaimana', 'adalah', 'ialah', 'itu', 'yang', 'kita', 'aku', 'saya', 'anda']
data_file = open('data.json', encoding='utf-8').read()
data = json.loads(data_file)


for intent in data['intents']:
    for pertanyaan in intent['patterns']:

        w = nltk.word_tokenize(pertanyaan)
        text_kata.extend(w)

        dataset.append((w, intent['tag']))


        if intent['tag'] not in clas:
            clas.append(intent['tag'])

text_kata = [lemmatizer.lemmatize(w.lower()) for w in text_kata if w not in ignoreWords]
text_kata = sorted(list(set(text_kata)))

clas = sorted(list(set(clas)))

print (len(dataset), "data")

print (len(clas), "classes", clas)

print (len(text_kata), "Kata Unik", text_kata)

pickle.dump(text_kata,open('text.pkl','wb'))
pickle.dump(clas,open('classes.pkl','wb'))

# Training Data
train_data = []
out_empty = [0] * len(clas)
for doc in dataset:

    bag = []

    kata_in_pattern = doc[0]
    kata_in_pattern = [lemmatizer.lemmatize(word.lower()) for word in kata_in_pattern]

    for w in text_kata:
        bag.append(1) if w in kata_in_pattern else bag.append(0)


    out_row = list(out_empty)
    out_row[clas.index(doc[1])] = 1

    train_data.append([bag, out_row])

random.shuffle(train_data)
train_data = np.array(train_data)
# create data train dan data test lists. X - patterns, Y - intents
x_train = list(train_data[:,0])
y_train = list(train_data[:,1])
print("Create Data Training")


# model dibuat menjadi 3 layers. layer 1. 140 node (neurons), layer 2 70 node (neurons) dan layer 3 output jumlah neurons
model = Sequential()
model.add(Dense(140, input_shape = (len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(70, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

# Mengcompile model
model.compile(optimizer="adam" , loss = 'MeanSquaredError' ,metrics = ['accuracy'])
#model.compile(optimizer="adam" , loss = 'CategoricalCrossentropy',metrics = ['accuracy'])
#train = model.fit(np.array(x_train), np.array(y_train), epochs=500, batch_size=10, verbose=2, validation_split=0.1)

#fitting dan simpan model 
train = model.fit(np.array(x_train), np.array(y_train), epochs=1000, batch_size=10, verbose=1, validation_split=0.1)
model.save('Model_Chat.h5', train)

Hasil_acc = model.evaluate(np.array(x_train), np.array(y_train), verbose=2)
print("Akurasi %s: %.2f%%" % (model.metrics_names[1], Hasil_acc[1]*100))

print("model created")