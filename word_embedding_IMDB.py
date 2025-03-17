#import lib
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
#veri seti yükleme
df=pd.read_csv("IMDB Dataset.csv")
documents=df["review"]

#metin temizleme
def clean_text(text):
    text=text.lower()
    text=re.sub(r"\d+","",text)#sayıları temizle
    text=re.sub(r"[^\w\s]","",text)#ozel karakterlri temizle 
    english_stopWords=set(stopwords.words('english'))
    text=" ".join([word for word in text.split()if  word not in english_stopWords])
    #text=simple_preprocess(text) desek assagıda ektradan yapmama gerek kalmazdı tokenize islemini
    return text 

cleaned_documents=[clean_text(row)for row in documents]
#metinleri tokenization
IMDB_tokenization=[simple_preprocess(doc)for doc in clened_documents]
#############################################################################################################################

#word2vec model tanımla
model=Word2Vec(sentences=IMDB_tokenization,vector_size=50,window=5,min_count=1,sg=0)
word_vectors=model.wv

words=list(word_vectors.index_to_key)
vectors=[word_vectors[word]for word in words]
#clustring KMeans K=2
#kümeleme yapan algoritma hepsini kümeliyo ve merkezden uzaklıgına göre ayarlıyor
kmeans=KMeans(n_clusters=2)
kmeans.fit(vectors)
clusters=kmeans.labels_

#PCA 50->2
pca=PCA(n_components=2)
reduced_vectors=pca.fit_transform(vectors)
#2 boyutlu gorsellestirme 
plt.figure()
plt.scatter(reduced_vectors[:,0], reduced_vectors[:,1],c=clusters,cmap="viridis")

centers=pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],c="red",marker="x",s=130,label="Center")
plt.legend()

#figure üzerine kelime eklenmesi
for i,word in enumerate(words):
    plt.text(reduced_vectors[i,0],reduced_vectors[i,1],word,fontsize=7)

plt.title("Word2Vec")
plt.show()
