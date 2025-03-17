#import lib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re 
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

df = pd.read_csv(r"C:\Users\OĞUZHAN ÇALIŞ\Desktop\NLP KURS\metinTemsili\spam.csv",encoding="Windows-1252")

#metin verilerini alalım
documents =df["v2"]
#pos yada negativ

#metin temizleme
def cleanText(text):
    #büyük kücük harf cevirimi
    text=text.lower()
    #rakamlari ve ozel karakterleri temizleme
    text=re.sub(r"[^a-z\s]","",text)
    #kisa anlamsız kelimeleri temizleme the in 
    textList=text.split()
    stopWordsEngilish=set(stopwords.words('english'))
    filterWord=[word for word in textList if word not in stopWordsEngilish]
    return " ".join(filterWord)#temizlenmis texti return et 
#metinleri  temizle
cleanedDoc=[cleanText(text) for text in documents]
#veri setinin içiri aktarmak
df = df.dropna(subset=["v2"])
#metin temizleme

# %% bow

#vectorize tanımla
vectorizer=CountVectorizer(min_df=50, max_df=0.9)
#metinleri -> sayısal hale getir
X=vectorizer.fit_transform(cleanedDoc)
#kelime kümesi göster
feature_names=vectorizer.get_feature_names_out()
#vektor temsili göster
vektor_temsili2=X.toarray() 
df_bow=pd.DataFrame(vektor_temsili2,columns=feature_names)

#kelime freakanslarını göster 
word_count=X.sum(axis=0).A1
wordFreq=dict(zip(feature_names,word_count))
df_word_freq = pd.DataFrame(wordFreq.items(), columns=['Kelime', 'Frekans']).sort_values(by="Frekans", ascending=True)

print(df_word_freq.to_string(index=False))
