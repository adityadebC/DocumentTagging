import re
from sklearn.feature_extraction.text import  TfidfVectorizer
import pandas as pd
import os
from collections import Counter
import math

path = "Train_docs"
train_tag_path = "Train_tags"


file_names = os.listdir(path)

full_corpus = []




def remove_string_special_characters(s):

    stripped = re.sub('[^a-zA-z\s]', '', s)
    stripped = re.sub('_', '', stripped)

    # Change any white space to one space
    stripped = re.sub('\s+', ' ', stripped)

    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped != '':
        return stripped.lower()
    
    
clean_text = []
for filename in file_names[0:]:
    with open(os.path.join(path,filename),'r') as f:
        text = f.readlines()
        for line in text:
            stripped = remove_string_special_characters(line)
            if stripped == None:
                continue

            clean_text.append(stripped)
        clean_text = ' '.join(clean_text)
        full_corpus.append(clean_text)
        clean_text = []


##Starting TFIDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, ngram_range=(1,2))
X = vectorizer.fit_transform(full_corpus)
features = (vectorizer.get_feature_names())


##Function to look for cosine similarity
def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

for i in range(0,len(X.toarray())):
    print(" this is for ", file_names[i])
    feature_index = X[i, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [X[i, x] for x in feature_index])
    data1 = []
    for w, s in [(features[i], s) for (i, s) in tfidf_scores]:
        data1.append([w, s])
    df = pd.DataFrame(data1, columns=['words','value'])
    df.sort_values(by=['value'], ascending=False, inplace=True)
    df = df.head(5)
    predicted_tags = df['words'].tolist()
    print(predicted_tags)
    
# Getting document similarity between tfidf predicted tags and actual tags

    with open(os.path.join(train_tag_path,('case'+file_names[i].split('_')[1]+'.txt'))) as f:
        txt = f.readlines()
    actual_tags = remove_string_special_characters(txt[0]).split()
    print(actual_tags)
    counter_predicted = Counter(predicted_tags)
    counter_actual = Counter(actual_tags)
    print(counter_cosine_similarity(counter_predicted, counter_actual) * 100)












