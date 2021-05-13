from django.shortcuts import render,get_object_or_404
import json
from .models import Product, Review
from django.views.generic import ListView
import spacy
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import string

def data(asin):
    """Create a list of common words to remove"""
    stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
                "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
                "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", 
                "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", 
                "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", 
                "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", 
                "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
                "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", 
                "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    """Load the pre-trained NLP model in spacy"""
    nlp=spacy.load("en_core_web_sm") 

    """Define a function to extract keywords"""
    def get_aspects(x):
        doc=nlp(x) ## Tokenize and extract grammatical components
        doc=[i.text for i in doc if i.text not in stop_words and i.pos_=="NOUN"] ## Remove common words and retain only nouns
        doc=list(map(lambda i: i.lower(),doc)) ## Normalize text to lower case
        doc=pd.Series(doc)
        doc=doc.value_counts().head(10).index.tolist() ## Get 10 most frequent nouns
        return doc

    reviews = Review.objects.filter(asin=asin).values_list('text',flat=True)
    rev = ''.join(reviews)

    return (get_aspects(rev))

def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens 

def encode_docs(tokenizer, max_length, docs):
    # integer encode
    encoded = tokenizer.texts_to_sequences(docs)
    # pad sequences
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded 

def predict_sentiment(reviews, vocab, tokenizer, max_length, model):
    sent_reviews=[]
    for review in reviews:
        line = clean_doc(review, vocab)
        padded = encode_docs(tokenizer, max_length, [line])
        yhat = model.predict(padded, verbose=0)
        percent_pos = yhat[0,0]
        if round(percent_pos) != 0:
            sent_reviews.append({'text':review,'score':(percent_pos),'sentiment':'NEGATIVE'})
        else:
            sent_reviews.append({'text':review,'score':(1-percent_pos),'sentiment':'POSITIVE'})
    return sent_reviews


# Create your views here.
class Index(ListView):
    template_name='index.html'
    model=Product
    context_object_name='products'
    paginate_by=20

def index(request):
    # with open('review\products.json') as pj:
    #     for line in pj:
    #         m=json.loads(line)
    #         prod=Product()
    #         prod.asin=m['asin']
    #         prod.title=m['title']
    #         prod.price=m['price']
    #         prod.save()

    # with open('review\\reviews.json') as rj:
    #     c=0
    #     for line in rj:
    #         try:
    #             if c%10000==0:
    #                 print(c)
    #             m=json.loads(line)
    #             rev=Review()
    #             rev.text=m['reviewText']
    #             rev.rating=m['overall']
    #             rev.reviewer=m['reviewerName']
    #             prod=Product.objects.filter(asin=m['asin'])[0]
    #             rev.asin=prod
    #             rev.save()
    #             c+=1
    #         except KeyError:
    #             continue
    return render(request,template_name='index.html')

def product(request,asin):
    if request.method=='GET':
        product=get_object_or_404(Product,asin=asin)
        reviews=Review.objects.filter(asin=asin)[:2]
        features=data(asin)
        count=Review.objects.filter(asin=asin).count()
        context={'product':product, 'reviews':reviews, 'count':count, 'features':features, 'analyse':False}
        return render(request,'product.html',context)
    if request.method=='POST':
        product=get_object_or_404(Product,asin=asin)
        feature=request.POST.get('feature')
        # reviews=Review.objects.filter(asin=asin)
        reviews = Review.objects.filter(asin=asin).values_list('text',flat=True)
        reviews=list(reviews)
        # count=len(reviews)
        reviews.insert(0,feature)
        ans=[]
        indexes=[]
        corpus_new=tuple(reviews)
        tfidf_v=TfidfVectorizer(norm="l2")
        tfid_mat=tfidf_v.fit_transform(corpus_new)
        temp=cosine_similarity(tfid_mat[0],  tfid_mat[1::])
        for i in range(len(temp[0])):
            ans.append(temp[0][i])
        for i in range(10):
            m = max(ans)
            indexes.append(ans.index(m))
            ans.remove(m)
        extracted_reviews=[]
        for index in indexes:
            extracted_reviews.append(reviews[index])
        print(extracted_reviews)
        features=data(asin)
        #vocab
        file = open(r'review\vocab.txt', 'r')
        vocab = file.read()
        file.close()
        vocab = set(vocab.split())
        #tokenizer
        dbfile = open(r'review\tokenizer.pickle', 'rb')     
        tokenizer = pickle.load(dbfile)
        #model
        max_length = 801
        model = load_model(r'review\model.h5') 

        sent_reviews = predict_sentiment(extracted_reviews, vocab, tokenizer, max_length, model)
        context={'product':product, 'reviews':sent_reviews, 'features':features, 'analyse':True}
        return render(request,'product.html',context)