# Script-TF-IDF
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(encoding='latin-1', ngram_range=(1, 1), tokenizer=None, analyzer = 'word', stop_words=stopwordplus)
countvec= count_vectorizer.fit_transform(data.Komentar).toarray()
countvec

countvec2 = pd.DataFrame(countvec)
countvec2

kata_kata = count_vectorizer.get_feature_names()
countvec3 = pd.DataFrame(countvec, columns=kata_kata)
countvec3

#Menghitung TF-IDF 
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=False, sublinear_tf=False)
tfidf = transformer.fit_transform(countvec)
tfidf

#Mengubah menjadi array
tfidf1 = tfidf.toarray()
tfidf1

tfidf2 = pd.DataFrame(tfidf1)
tfidf2

kata_kata2 = count_vectorizer.get_feature_names()
df1 = pd.DataFrame(tfidf1, columns=kata_kata2)
df1

