from flask import Flask, request, render_template
import pickle
import pandas as pd
import nltk
import re
from string import punctuation
import xgboost as xgb

app = Flask(__name__, template_folder="templates")

# Load the model
model = pickle.load(open('bst.sav','rb'))
vectorizer_rvdesc = pickle.load(open('vectorizer_rvdesc.pkl','rb'))
label_encoder = pickle.load(open('label_encoder.pkl','rb'))

# Function to clean text data
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))

# Punctuation pattern
p = re.compile(r'[\s{}0-9]+'.format(re.escape(punctuation)))

def cleantexts(df,punctuation_patterns,col):
    df[col] = df[col].apply(lambda x: ' '.join([word.lower() for word in nltk.word_tokenize(x)]))
    df[col] = df[col].apply(lambda x:' '.join([ i for i in nltk.word_tokenize(x) if i not in stop_words and len(i)>1])) #and not i.isnumeric() and not i.isalpha()
    df[col] = df[col].apply(lambda x: re.sub(punctuation_patterns,' ',x))
    return df 

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data = request.form.get('text')
        # Make prediction
        df = pd.DataFrame([str(data)], columns=['review_description'])
        df = cleantexts(df=df,punctuation_patterns=p,col='review_description')
        x_test = vectorizer_rvdesc.transform(df['review_description'])
        feature_desc_test = pd.DataFrame(x_test.toarray())
        columns_test = [vectorizer_rvdesc.get_feature_names()]
        feature_desc_test.columns = columns_test
        dtest = xgb.DMatrix(data=feature_desc_test)
        pred = model.predict(dtest)
        pred = [int(i) for i in pred.tolist()]
        pred = label_encoder.inverse_transform(pred)
        pred = pred.tolist()
        return render_template('index.html', variety=pred[0])
    return render_template('index.html', variety='')
    
if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=3000, debug=True)