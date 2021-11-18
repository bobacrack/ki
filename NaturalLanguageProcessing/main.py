import csv
from words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer

porter = PorterStemmer()
lemma = WordNetLemmatizer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(lemma.lemmatize(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def writeCSV2(reviews, res, features):
    with open('NaturalLanguageProcessing/data/F500_22.csv', 'w') as f:
        writer = csv.writer(f)
        for feature in features:
            f.write(feature + ",")
        f.write("evaluation\n")
        for i in range(len(reviews)):
            row = ""
            for el in reviews[i]:
                row += str(el) + ","
            row += res[i] + "\n"
            f.write(row)


def prepData():
    reviews = []
    res = []
    with(open("NaturalLanguageProcessing/data/KI-imdb-sentiment-2011.csv", "r", encoding="UTF-8")) as fin:
        for line in fin.readlines():
            text = line.strip()
            listDoc = text.split(",")
            listDoc[0].replace("'", "")
            #listDoc[0] = stemSentence(listDoc[0])
            reviews.append(listDoc[0])
            res.append(listDoc[1])

    del reviews[0]
    del res[0]
    return reviews, res




reviews, result = prepData()
countVectorizer = CountVectorizer(analyzer='word', max_features=500, stop_words=ENGLISH_STOP_WORDS, ngram_range=(2,2))
X = countVectorizer.fit_transform(reviews)
features = countVectorizer.get_feature_names_out()
xresult = X.toarray()
print(xresult[0])
writeCSV2(xresult, result, features)
#print(result)

