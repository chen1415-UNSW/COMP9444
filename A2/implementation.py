import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import re

batch_size = 50

# words need to be removed from input reviews. They are meaningless.

stop_word=(
'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 
'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 
'cant', 'co', 'computer', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 
'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 
'hereby', 'herein', 'hereupon', 'hers', 'herse"', 'him', 'himse"', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itse"', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 
'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myse"', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 
'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 
'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 
'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 
'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 
'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'
)

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    
    data = []
    
    # extract data from review.tar.gz first
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'review/')):
        with tarfile.open('reviews.tar.gz', "r") as tarball:
            dir = os.path.dirname(__file__)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tarball, os.path.join(dir,"review/"))
    
    # open pos file
    for file in glob.glob(r'review/pos/*.txt'):
        with open(file, 'r', encoding="utf-8") as f:
            word_arr = f.readline().split(" ")
            word_pos_vector = [0 for i in range(40)]
            index = 0
            for word in word_arr:
                if index > 39:
                    break
                
                word = word.lower()
                word = re.sub(r'^\W*|\W*$', '', word)
                if word != '' and word not in stop_word:
                    word_pos_vector[index] = glove_dict.get(word, 0)
                    #print("{} : {}".format(word,glove_dict.get(word, 0)))
                index += 1
                
            data.append(word_pos_vector)
                
    # open neg file
    for file in glob.glob(r'review/neg/*.txt'):
        with open(file, 'r', encoding="utf-8") as f:
            word_arr = f.readline().split(" ")
            word_neg_vector = [0 for i in range(40)]
            index = 0
            for word in word_arr:
                if index > 39:
                    break
                
                word = word.lower()
                word = re.sub(r'^\W*|\W*$', '', word) 
                if word != '' and word not in stop_word:
                    word_neg_vector[index] = glove_dict.get(word, 0)
                    #print("{} : {}".format(word,glove_dict.get(word, 0)))
                index += 1
                
            data.append(word_neg_vector)
    
    return np.array(data)


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119}
    """
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    
    embeddings = []
    word_index_dict = {"UNK":0}
    zero_vector = [0 for i in range(50)]
    # first element is zero vector for all unknown words
    embeddings.append(zero_vector)
    # then index start from 1
    index = 1
    for line in data.readlines():
        word_arr = line.strip().split(" ")
        # first word is the string of word
        word_name = word_arr[0]
        # the rest are number vector
        word_vector = list(map(float, word_arr[1:]))
        
        word_index_dict[word_name] = index
        embeddings.append(word_vector)
        # marking index
        index += 1
    
    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""
    
    tf.reset_default_graph()
    lstm_size = 30
    lstm_layers = 1
    learning_rate = 0.001
    
    n_input = 50
    n_steps = 40
    n_classes = 2
    n_hidden = lstm_size
    
    weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], stddev=0.1)),
    'in': tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.1))
    }
    biases = {
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,])),
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden,]))
    }
    
    dropout_keep_prob = tf.placeholder_with_default(0.5, shape=(), name="dropout_keep_prob")
    
    input_data = tf.placeholder(tf.int32, [batch_size, 40], name="input_data")
    labels = tf.placeholder(tf.float32, [batch_size, 2], name="labels")
    
    embeddings = tf.convert_to_tensor(glove_embeddings_arr)
    input_vector = tf.nn.embedding_lookup(embeddings, input_data)
    # input X => (batch_size, 40, 50)
    # first input layer
    X = tf.reshape(input_vector, [-1, 50])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # reshape to input to lstm 
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden])
    
    lstmCell = rnn.BasicLSTMCell(lstm_size)
    lstmCell_dropout = rnn.DropoutWrapper(cell=lstmCell, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob)
    #init_state = lstmCell.zero_state(batch_size, dtype=tf.float32)
    
    outputs, state = tf.nn.dynamic_rnn(lstmCell_dropout, X_in, dtype=tf.float32)
   
    #weight = tf.Variable(tf.truncated_normal([lstm_size, 2]))
    #bias = tf.Variable(tf.constant(0.1, shape=[2]))
  
   
    #prediction = tf.matmul(outputs[-1], weights['out']) + biases['out']
    outputs = tf.transpose(outputs, [1, 0, 2])
    value = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
    prediction = tf.matmul(value, weights['out']) + biases['out']
    
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name="accuracy")
    loss = tf.reduce_mean(
           tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), 
           name="loss"
           )
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss

    