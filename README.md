# DocumentTagging
I got a bunch of Train_docs and Train_tags. The tags are of variable number for each train_doc. Here are the approached I thought of:

Approach 1: Using LexNLP Package. 
            This is a very specialized NLP package for legal documents. But it has certain limitations. It runs only in python 3.6.
			Also most of the functions requires in-built dictionaries which are specialized for US courts and legal proceedings.
			So, I decided not to go with it. 
			
Approcah 2: Using Neural Networks/Deep Learning
            I couldn't use the training docs and tags as train data and labels because I wasn't sure how to accomodate variable number of tags in training.
			I thought of converting the data using Tensorflow Tokenizer.texts_to_sequences and converting the labels also like that. But I couldn't think of a way to train using variable sequence numbers for labels.
			I will try to figure out how to do it if at all it is possible
			
Approcah 3: Using TF-IDF
      I applied TFIDF using TFIDFVectorizer.
			I got the predicted tags from there.
			I looked for cosine similarity between the predicted tags and actual tags. Based on that I modified the TFIDF parameters.
			
			After that I simply used that TFIDF parameters and generated a folder with the predicted tags for the test_docs
			
			Given time, we can further tune the TFIDF parameters and put in some more stopwords list and tune other hyperparameters to get more accurate results.
			
			I have gone with 5 tags for each document. We can make that variable as well by making sure that tags which falls under a certaing IDF number comes to the tags document.
			
			The TestingScript.py is the one that created the tags for the test_docs
			
			
			
