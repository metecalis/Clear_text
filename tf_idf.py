#import lib
from xml.dom.minidom import Document
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
#onrek olustur
document=[
    "köpek cok tatli bir hayvan",
    "kedi cok tatli bir hayvan",
    "inek süt veren bir hayvan"
]
#vector tanımla
tfidf_vectorizer=TfidfVectorizer()
#metinleri sayısal hale cevir
X=tfidf_vectorizer.fit_transform(document)
#kelime kümesini incele
featureNames=tfidf_vectorizer.get_feature_names_out() #sözel seyleri cantaya aktardır herseyi
print(featureNames)
#vector temsilini incele 
vektor_temsili=X.toarray()
df_tfidf=pd.DataFrame(vektor_temsili,columns=featureNames)
print(df_tfidf)
#ortalama tf idf degerlerine bak
tf_idf=df_tfidf.mean(axis=0)