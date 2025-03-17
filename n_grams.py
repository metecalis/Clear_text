#import lib
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


#örnek metin
df=pd.read_csv("IMDB Dataset.csv")
documents=df["review"]

def clean_text(text):
    text=text.lower()
    text=re.sub(r"^\d+","",text)
    text=re.sub(r"^\w\s","",text)
    text = re.sub(r"[^\x00-\x7F]+", "", text) 
    text = re.sub(r"(.)\1{2,}", "", text)
    text = re.sub(r"(\d)\1{2,}", r"\1", text)
    englishStopWords=set(stopwords.words('english'))
    text=" ".join([word for word in text.split() if word not in englishStopWords and len(word)>2])
    return text
clened_documents=[clean_text(row)for row in documents]

# unigram bigram trigram seklinde 3 farklı n degerine sahip gram modeli 
vectorizer_unigram=CountVectorizer(ngram_range=(1,1),max_df=0.95, min_df=10,stop_words='english')
vectorizer_bigram=CountVectorizer(ngram_range=(2,2),max_df=0.95, min_df=10,stop_words='english')
vectorizer_trigram=CountVectorizer(ngram_range=(3,3),max_df=0.95, min_df=10,stop_words='english')
#trigram_features = [gram for gram in trigram_features if "br br" not in gram]
#unigram
X_unigram=vectorizer_unigram.fit_transform(clened_documents)
unigram_features=vectorizer_unigram.get_feature_names_out()
#bigram
X_bigram=vectorizer_bigram.fit_transform(clened_documents)
bigram_features=vectorizer_bigram.get_feature_names_out()
 #trigram
X_trigram=vectorizer_trigram.fit_transform(clened_documents)
trigram_features=vectorizer_trigram.get_feature_names_out()

#sonucları incele 
print(f"unigram features {unigram_features}")
print(f"bigram features {bigram_features}")
print(f"trigram features {trigram_features}")