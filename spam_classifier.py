import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    count=0
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)
        count += 1

    # print(count)
    return DataFrame(rows, index=index)

def getExamples(path):
    ret = []
    for _, message in readFiles(path):
        ret.append(message)
        
    return ret
        
        
trainData = DataFrame({'message': [], 'class': []})

trainData = trainData.append(dataFrameFromDirectory('emails/spam', 'spam')[:2000])
trainData = trainData.append(dataFrameFromDirectory('emails/ham', 'ham')[:2000])


testSpam = getExamples('emails/spam')[-500:]
testHam = getExamples('emails/ham')[-500:]


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(trainData['message'].values)

classifier = MultinomialNB()
targets = trainData['class'].values
classifier.fit(counts, targets)


# Simple test
examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?", "Increase your finger size fast!", "Whatsup?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)

# Spam test
spamCounts = vectorizer.transform(testSpam)
predictions = classifier.predict(spamCounts)
unique, counts = numpy.unique(predictions, return_counts=True)
print(dict(zip(unique, counts)))

# Ham test
hamCounts = vectorizer.transform(testHam)
predictions = classifier.predict(hamCounts)
unique, counts = numpy.unique(predictions, return_counts=True)
print(dict(zip(unique, counts)))