import pickle
import numpy as np
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from keras.models import load_model
model = load_model('Model_Chat.h5')
intents = json.loads(open('data.json', encoding='utf-8').read())
text_kata = pickle.load(open('text.pkl','rb'))
clas = pickle.load(open('classes.pkl','rb'))


def clean_sentence(sentence):
    kata_sentence = nltk.word_tokenize(sentence)
    kata_sentence = [lemmatizer.lemmatize(word.lower()) for word in kata_sentence]
    return kata_sentence

# return bag of words array: 0 atau 1 untuk setiap kata yang ada dalam kalimat

def bag_of_word(sentence, text_kata, details=True):
    # tokenize pattern
    kata_sentence = clean_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(text_kata)
    for s in kata_sentence:
        for i,w in enumerate(text_kata):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def prediction(sentence, model):
    # filter out prediksi
    p = bag_of_word(sentence, text_kata, details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    # mengurutkan peluang kata yang mirip (probability)
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": clas[r[0]], "probability": str(r[1])})
    return return_list

def getresponsebot(ints, intents_json):
    tags = ints[0]['intent']
    list_intents = intents_json['intents']
    for i in list_intents:
        if(i['tag']== tags):
            result = random.choice(i['responses'])
            break
        else:
            result = "Maaf, Bisa Tolong Ajukan pertanyaan yang benar!!"
    return result

def botResponse(answers):
    ints = prediction(answers, model)
    res = getresponsebot(ints, intents)
    return res


from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
app.static_folder = 'static'


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/artikel")
def artikel():
    return render_template("artikel.html")


@app.route('/predict', methods=["GET", "POST"])
def predict():
    text = request.get_json().get("message")
    response = botResponse(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == '__main__':
    app.run()