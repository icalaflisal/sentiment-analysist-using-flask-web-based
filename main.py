from flask import Flask, request, render_template, Response
import os
import pandas as pd
from pandas import DataFrame
import sklearn
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
pd.set_option('display.max_colwidth', 2500)
pd.set_option('display.max_rows', 5)
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)

#Import Dataset 
data = pd.read_csv("data_twitter.csv",encoding="ISO-8859-1")
label = pd.read_csv("labeling.csv",encoding="ISO-8859-1")
train= pd.concat([data,label], axis=1)
test_pd = pd.DataFrame(train)
x_train = np.array(train['text'])
y_train = np.array(train['label'])

#Split Data into Test and Train make Test Data 30% and Train Data 70%
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=13)

#Create Train Data and Test Data
d_train = pd.DataFrame()
d_train['trainx'] = Train_X
d_train['trainy'] = Train_Y

d_test = pd.DataFrame()
d_test['testx'] = Test_X
d_test['testy'] = Test_Y

#Create Variable for Train dan Test for "text"
xtrain = d_train['trainx']
xtest = d_test['testx']

#Cleansing Data & Case Folding
def cleansing(text):
    clean_data = []
    for x in (text[:]):

        #Cleansing Data
        new_text = re.sub('<.*?>', '', x)   # remove HTML tags
        new_text = re.sub(r'[^\w\s]', '', new_text) # remove punc.
        new_text = re.sub(r'\d+','',new_text)# remove numbers

        #Case Folding
        new_text = new_text.lower() # lower case, .upper() for upper          
        if new_text != '':
            clean_data.append(new_text)
    return clean_data

#Case Folding variable Train & Test
casefolding_trainx = cleansing(xtrain)
casefolding_tesx = cleansing(xtest)

#Create Column Case Folding Train & Test
d_train['case_folding'] = casefolding_trainx
d_test['case_folding'] = casefolding_tesx

#Tokenization
def identify_tokens(row):
    review = row['case_folding']
    tokens = word_tokenize(review)
    # Punctuation not taken
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

#Tokenization Variable Train and Test
#Train Tokenization
d_train['token'] = d_train.apply(identify_tokens,axis=1)
tokens_trainx = d_train['token']

#Test Tokenization
d_test['token'] = d_test.apply(identify_tokens,axis=1)
tokens_testx = d_test['token']

#Stopword Removal
stops = set(stopwords.words("indonesian"))
def remove_stops(row):
    my_list = row['token']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

#Stopword Variable Train and Test
#Train Stopword
d_train['stopwords'] = d_train.apply(remove_stops, axis=1)
stopword_trainx=d_train['stopwords']

#Test Stopword
d_test['stopwords'] = d_test.apply(remove_stops, axis=1)
stopword_testx=d_test['stopwords']

#Stemming
def stem_list(row):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    my_list = row['stopwords']
    stemmed_list = [stemmer.stem(word) for word in my_list]
    return (stemmed_list)

#Stemming Variable Train and Test
#Train Stemming
d_train['stemming'] = d_train.apply(stem_list, axis=1)
stem_trainx=d_train['stemming']
d_train['final']=stem_trainx.astype(str) #array list ke string
preprocessing_trainx = d_train['final'] #final column

#Test Stemming
d_test['stemming'] = d_test.apply(stem_list, axis=1)
stem_testx=d_test['stemming']
d_test['final']=stem_testx.astype(str) #array list ke string
preprocessing_testx = d_test['final'] #final column

#Data train and test for "label"
preprocessing_trainy = d_train['trainy']
preprocessing_testy = d_test['testy']

#Machine Learning Model using Multinominal Naive Bayes
model = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB()) ])

model.fit(preprocessing_trainx, preprocessing_trainy) #Train the Model
predicted = model.predict(preprocessing_testx) # Predict the test cases
d_test['prediksi'] = predicted #Create Column Prediction Result

#Accuracy
akurasi = (accuracy_score(preprocessing_testy, predicted)*100)
kesalahan= 100-(akurasi)
print (akurasi)

@app.route("/")
@app.route("/index")

def index():
    trainNew = test_pd
    label = test_pd['label']
    pos = label[label==1]
    neg = label[label==-1]
    net = label[label==0]
    return render_template('index.html', tables=[trainNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')])

@app.route("/casefolding")
def casedata():
    trainNewA = pd.Series(xtest, name='Data Test').reset_index()
    trainNewB = pd.Series(casefolding_tesx, name='Case Folding').reset_index()
    trainNew = pd.concat([trainNewA,trainNewB],axis=1)
    trainNew = trainNew.drop("index", axis=1)
    return render_template('casefolding.html',  tables=[trainNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')])

@app.route("/token")
def tokendata():
    trainNewA = pd.Series(xtest, name='Data Test').reset_index()
    trainNewB = pd.Series(tokens_testx, name='Tokenization').reset_index()
    trainNew = pd.concat([trainNewA,trainNewB],axis=1)
    trainNew = trainNew.drop("index", axis=1)
    return render_template('tokenization.html',  tables=[trainNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')])

@app.route("/stopwords")
def stopwordsdata():
    trainNewA = pd.Series(xtest, name='Data Test').reset_index()
    trainNewB = pd.Series(stopword_testx, name='Stop Words Removal').reset_index()
    trainNew = pd.concat([trainNewA,trainNewB],axis=1)
    trainNew = trainNew.drop("index", axis=1)
    return render_template('stopword.html',  tables=[trainNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')])

@app.route("/stemming")
def stemmingdata():
    trainNewA = pd.Series(xtest, name='Data Test').reset_index()
    trainNewB = pd.Series(stem_testx, name='Stemming').reset_index()
    trainNew = pd.concat([trainNewA,trainNewB],axis=1)
    trainNew = trainNew.drop("index", axis=1)
    return render_template('stemming.html',  tables=[trainNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')])

@app.route("/sentiment")
def hasilsentiment():
    trainNewA = pd.Series(xtest, name='Data Test').reset_index()
    trainNewB = pd.Series(predicted, name='Hasil Sentimen').reset_index()
    trainNew = pd.concat([trainNewA,trainNewB],axis=1)
    trainNew = trainNew.drop("index", axis=1)
    return render_template('hasil.html', tables=[trainNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')])

@app.route("/statistiksentimen")
def akurasisentimen():
    trainNewA = pd.Series(xtest, name='Data Test').reset_index()
    trainNewB = pd.Series(predicted, name='Prediction').reset_index()
    trainNew = pd.concat([trainNewA,trainNewB],axis=1)
    trainNew = trainNew.drop("index", axis=1)
    label = d_test['prediksi']
    colors = [ "#00008B","	#DC143C"]
    return render_template('akurasisentimen.html', akurasi=akurasi, kesalahan=kesalahan, tables=[trainNew.to_html(classes='table table-hover table-bordered',header='true',justify='justify',table_id='tabel')], set=zip(colors))

if __name__ == '__main__':
	app.run(debug=True)