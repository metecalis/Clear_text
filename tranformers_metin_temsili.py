from transformers import AutoTokenizer,AutoModel
import torch
#model ve tokenizer indir
model_name="bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModel.from_pretrained(model_name)
#input text tanımla
texts = [
    "Transformers can be used for natural language processing.",
    "BERT is a powerful model for understanding language.",
    "Artificial intelligence is revolutionizing industries.",
    "I love learning new things in machine learning.",
    "This is a great example of using BERT for NLP tasks."
]
#metinleri token
#inputs=tokenizer(text,return_tensors="pt") #tokenlastırdan sonra çıktıyı pytorch tensoru olarak return eden
#modeli kullanarak metin temsili olustur
for text in texts:
    print(f"Metin: {text}")

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad(): #gradyanların hesaplanması durdurulur bölecek bellek daha verimli kullanılır
        outputs=model(**inputs)    
#modelin çıkısından son gizli drumu alalım
    last_hidden_state=outputs.last_hidden_state #tum token çıktılarını almak için 
#ilk tokenin embedding inin alalım ve print edelim
    first_token_embedding=last_hidden_state[0,0,:].numpy()
    print(f"Metin temsili (ilk token): {first_token_embedding[:10]}...")  # İlk 10 değeri gösterelim
    print("-" * 100)  # Ayrım çizgisi

