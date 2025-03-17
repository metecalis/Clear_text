#count vectorizeri içeriye aktar
from sklearn.feature_extraction.text import CountVectorizer

#kendi veir setini olustur
documents=[
    "kedi bahçede",
    "kedi evde"]
#vectorizeri tanımla
vectorizer=CountVectorizer()
#metni sayısal yöntemlere çevir
X= vectorizer.fit_transform(documents) #fit et ve dönüstür

# kelime kümesi olustruma [bahçede,kedi,evde]

featureNames=vectorizer.get_feature_names_out() #kelime kümesini olusturma 
print(f"kelime kümesi :{featureNames}")
# vektor temsili 
vectorTemsili=X.toarray()

print(f"kelime kümesi :{vectorTemsili}")