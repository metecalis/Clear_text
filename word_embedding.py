""""
word2vec(google)
fasttext(facebook)
"""
#import lab
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#princle  componenet analys verideki boyutları azaltarak en iyi varsyansı açıklayan yeni dogrusal yöntemler kullanılrı
from gensim.models import Word2Vec,FastText
from gensim.utils import simple_preprocess
#örnek verfi seti
sentences = [
    # Hayvanlar ve Özellikleri
    "köpek çok tatlı bir hayvandır",
    "kedi evcil bir hayvandır",
    "köpekler evcil veya tehlikeli olabilir",
    "köpekler sahiplerine sadıktır",
    "kediler özgür ruhludur",
    "hayvanlar insanların en iyi dostudur",
    "hayvanlar insanlara sevgi gösterir",
    "köpekler insanlarla güçlü bir bağ kurar",
    "kediler yalnız kalmayı sever",
    "evcil hayvanlar insanlara huzur verir",
    
    # İnsanlar ve Sosyal Bağlar
    "insanlar sosyal varlıklardır",
    "insanlar ve hayvanlar arasında güçlü bir bağ vardır",
    "insanlar birlikte yaşamaktan mutluluk duyar",
    "dostluk insanları bir arada tutar",
    "arkadaşlık güven ve sadakat gerektirir",
    "dostlar birbirine her zaman destek olmalıdır",
    "insanlar birbirine sevgi ve saygı göstermelidir",
    "aile her zaman insanın yanında olur",
    "mutluluk paylaşınca çoğalır",
    "güven bir ilişkinin temelidir",
    
    # Duygular ve İlişkiler
    "sadakat dostluk ve sevgi önemlidir",
    "mutlu olmak için sevgi ve dostluk gerekir",
    "üzgün olduğunda arkadaşların seni destekler",
    "yalnızlık insanı hüzünlendirir",
    "hayat bazen zordur ama dostluk her şeyi güzelleştirir",
    "sevgi en güçlü duygulardan biridir",
    "üzüntü ve mutluluk hayatın bir parçasıdır",
    "gerçek dostluk zor zamanlarda belli olur",
    "sevgi dolu insanlar her zaman mutlu olur",
    "insanlar bazen yalnız hissetse de dostları onları unutmamalıdır",
    
    # Günlük Hayattan Cümleler
    "güzel bir sabah kahvesi insanı mutlu eder",
    "yağmurlu günlerde kitap okumak huzur verir",
    "iyi bir sohbet insanın ruhunu dinlendirir",
    "güneşli bir gün moralimizi yükseltir",
    "geceleri yıldızlara bakmak insanı rahatlatır",
    "sessiz bir ortam insanı huzurlu hissettirir",
    "insanların birbirine gülümsemesi küçük ama önemli bir şeydir",
    "bir fincan çay yorgunluğu alır",
    "müzik insan ruhuna iyi gelir",
    "sanat insanlara ilham verir"
]

tokenized_sentences=[simple_preprocess(sentence) for sentence in sentences ]#token haliine getirdik isledik sentecleri kücük harflere çevirip token haline getirdik
#yaptıgımız sey önce cümleleri token haline getirdik sonra w2 ve fasttext modelyle 50 boyutlu bir vector haline getirdik
#50 vectorü görselletirecemiz için  sonra PCA ile 3 boyutlu bir sekilde görsellestirkdik 
#karsımıza çık görsellere baktıgımızda  hayvanlar ve hayvnalardır yakın bir sekilde çıktı yada dost ve arkdas yakın çıktı 
#word2vec
word2_vec_model=Word2Vec(sentences=tokenized_sentences,vector_size=50,window=5,min_count=1,sg=0)
#fasttext
fast_text_model=FastText(sentences=tokenized_sentences,vector_size=50,window=5,min_count=1,sg=0)
#görsellestirme teknigi
def plot_word_embedding(model,title):
    word_vectors=model.wv
    words=list(word_vectors.index_to_key)
    vectors=[word_vectors[word]for word in words]
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(vectors)
    #pca
    pca=PCA(n_components=3)
    reduced_vector=pca.fit_transform(vectors)
    #3d görsel
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection="3d")
    #vectorleri ciz
    ax.scatter(reduced_vector[:, 0], reduced_vector[:, 1], reduced_vector[:, 2], c=labels, cmap="viridis")
    #kelimeleri etikle
    for i,word in enumerate(words):
        ax.text(reduced_vector[i,0], reduced_vector[i,1], reduced_vector[i,2], word,fontsize=12)

    ax.set_title(title)
    ax.set_xlabel("Componenet 1")
    ax.set_ylabel("Componenet 2")    
    ax.set_zlabel("Componenet 3")
plot_word_embedding(word2_vec_model,"Word2Vec")
plot_word_embedding(fast_text_model,"fasttext")     
plt.show()   
