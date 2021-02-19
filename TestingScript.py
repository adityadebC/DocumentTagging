import re
from sklearn.feature_extraction.text import  TfidfVectorizer
import pandas as pd
import os
import csv

path = "Test_docs"
test_tag_path = "Test_tags"

if not os.path.exists(test_tag_path):
    os.mkdir(test_tag_path)

file_names = os.listdir(path)
# file_names = file_names[0:4]

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
    with open(os.path.join(path, filename), 'r') as f:
        text = f.readlines()
        for line in text:
            stripped = remove_string_special_characters(line)
            if stripped == None:
                continue

            clean_text.append(stripped)
        clean_text = ' '.join(clean_text)
        full_corpus.append(clean_text)
        clean_text = []

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, ngram_range=(1, 2))
X = vectorizer.fit_transform(full_corpus)
features = (vectorizer.get_feature_names())
#

for i in range(0, len(X.toarray())):
    print(" this is for ", file_names[i])
    feature_index = X[i, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [X[i, x] for x in feature_index])
    data1 = []
    for w, s in [(features[i], s) for (i, s) in tfidf_scores]:
        data1.append([w, s])
    df = pd.DataFrame(data1, columns=['words', 'value'])
    df.sort_values(by=['value'], ascending=False, inplace=True)
    df = df.head(5)
    list_of_tags = df['words'].tolist()
    TagFileName = 'case'+file_names[i].split('_')[1]+'.csv'
    with open(os.path.join(test_tag_path, TagFileName),'w') as file:
        writer = csv.writer(file)
        writer.writerow(list_of_tags)

















