from sklearn.feature_extraction.text import CountVectorizer
import pickle

corpus_file = "environment/tokenization/question_corpus_mlp.txt"
preprocessor_file = "preprocessing/count_tokenizer.pkl"
with open(corpus_file, "r") as f:
    corpus = f.readlines()
corpus = [line.strip() for line in corpus]

vectorizer = CountVectorizer(token_pattern=r"\$?(?u)\b\w\w+\b") #Add $ symbol to token pattern to identify objects
X = vectorizer.fit(corpus)

with open(preprocessor_file, 'wb') as f:
    pickle.dump(vectorizer, f)