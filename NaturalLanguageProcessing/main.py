import csv
import re
from sklearn.feature_extraction.text import CountVectorizer

illegal = ['and','zombie']

def write(result):
    with open("NaturalLanguageProcessing/data/vector.csv", "w", encoding="UTF-8") as f:
        writer = csv.writer(f)
        writer.writerow(result)


def prepData():
    reviews = []
    res = []
    with(open("NaturalLanguageProcessing/data/KI-imdb-sentiment-2011.csv", "r", encoding="UTF-8")) as fin:
        for line in fin.readlines():
            text = line.strip()
            listDoc = text.split(",")
            reviews.append(listDoc[0].replace("'", ""))        
            res.append(listDoc[1])

    del reviews[0]
    del res[0]
    return reviews, res


reviews, result = prepData()
countVectorizer = CountVectorizer(analyzer='word', max_features=100)
X = countVectorizer.fit_transform(reviews)
#print(X)
print(countVectorizer.get_feature_names_out())
result = X.toarray()
#write(result)
print(result)

