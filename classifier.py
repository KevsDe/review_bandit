import pandas as pd
import src.text_analysis as ta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pickle


print('Loading the dataset')
amz_rev = pd.read_csv('output/amz_rev_final.csv')
rename = {0:'Real',1:'Spam'}
amz_rev['low_quality'] = amz_rev['low_quality'].replace(rename)

print('Vectorizing')
tfidf=TfidfVectorizer(min_df=.02, max_df =0.08, tokenizer=ta.spacy_tokenizer)

X = tfidf.fit_transform(amz_rev['reviewText'].values.astype('U'))

print('Training Model')
model = SVC()
model.fit(X, amz_rev['low_quality'].values)

print('Calculating the K-fold cross validation')
validation = cross_val_score(model, X, amz_rev['low_quality'].values, cv=5)
print(f'K-fold cross validation Accuracy: {round(validation.mean() * 100,2)}%')


filename = 'output/prediction_model'
pickle.dump(model, open(filename,'wb'))

print('The model has been successfully saved in the directory output as prediction_model')

