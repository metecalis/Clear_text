#import lib
from pydoc import TextRepr, text
from numpy import vectorize
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
#veri seti yükle
df=pd.read_csv(r"C:\Users\OĞUZHAN ÇALIŞ\Desktop\NLP KURS\metinTemsili\spam.csv",encoding="Windows-1252")
print=df.head(10)
document=df["v2"]
#veri temizleme blogu 
def cleanText(text):
    text=text.lower()
    #texti re kütüphanesinden sub özelligiyle adan z harici olanları temizledik!gibi
    text=re.sub(r"[^a-z\s]","",text)
    #texti böldükki filtered wordde kıyaslama yapalım egerki text stopwordsin icinde yoksa cevirdik 
    textList=text.split()
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords")
    stopWordsEnglish=set(stopwords.words('english'))
    filterWord=[text for text in textList if text not in stopWordsEnglish]
    return " ".join(filterWord)
#tf idf
cleanedDoc = [cleanText(text) for text in df["v2"].astype(str)]

vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(cleanedDoc)

#kelime kümesini incele
feature_names=vectorizer.get_feature_names_out()
tfidf_score=X.mean(axis=0).A1 #her kelimenin ortalama tfidf degeri bul
#tf idf skorlarını içeren bir df olsutur
df_tfidf=pd.DataFrame({"word":feature_names,"tfidf_score":tfidf_score})#kelime ve kelimelere karsılık gelen seyleri görüyoruz 
#skorlara göre sırala sonucları incele ,
df_tfidf_sorted=df_tfidf.sort_values(by="tfidf_score",ascending=False)
print(df_tfidf_sorted.head(10))