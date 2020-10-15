from flask import Flask, render_template, request, Markup
import os
from src.amz_scraping import scrape, reviews_extractor
from src.text_analysis import spacy_tokenizer
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# necesario en pythonanywhere
PATH=os.path.dirname(os.path.abspath(__file__))

# flask app
app=Flask(__name__)


@app.route("/")
def my_form():
    return render_template('form.html')

# main app
@app.route("/", methods=['POST', 'GET'])
def main():
    print('loading amz_rev')
    amz_rev = pd.read_csv('output/amz_rev_final.csv')
    rename = {0:'Real',1:'Spam'}
    amz_rev['low_quality'] = amz_rev['low_quality'].replace(rename)
    print('vectorizing')
    tfidf=TfidfVectorizer(min_df=.02, max_df =0.08, tokenizer=spacy_tokenizer)
    X = tfidf.fit_transform(amz_rev['reviewText'].values.astype('U'))
    print('pickle')
    filename = 'output/prediction_model'
    loaded_model = pickle.load(open(filename,'rb'))
    print('request')
    text = request.form['URL']
    df = pd.DataFrame(data=reviews_extractor(scrape(text)))
    print('analysis')
    def reviews_quality(review):
        vec_rev = tfidf.transform([review])
        predict = loaded_model.predict(vec_rev)
        return predict[0]
    checking = []
    num_review = []
    for idx in range(len(df['Reviews'])):
        checking.append(reviews_quality(df['Reviews'][idx]))
        num_review.append(idx+1)
    dicto = {'Review': num_review , 'Conclusion': checking}
    ended = pd.DataFrame(data=dicto)
    return render_template('simple.html',  tables=[ended.to_html(classes='data', header="true")], titles=ended.columns.values)


if __name__=='__main__':
    app.run(debug=True)