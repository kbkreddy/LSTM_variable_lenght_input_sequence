# MLP for the IMDB problem
import numpy
from keras.models import load_model
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
import pandas as pd
#from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)




def batch_iter(data, labels,num_batches_per_epoch, shuffle=False):
  #  num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels
            i=0    
            while i< data_size-2 : 
                start_index = i
                while (len(x[i]) == len(x[i+1])) :
                        i=i+1
                end_index = i
                i=i+1
                X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
                yield X, y

    return num_batches_per_epoch, data_generator()



def step_count(x) :
	
	x_train=x
	x_train.to_csv('save.csv',header=["data"])

	read_data=pd.read_csv('save.csv')             
                             
	x_test=read_data.data 
      
	



	total=x_test.shape[0]
	print(total)



	i=0
	count=0
	size=[]

 


	while i<=total-2 :
        

        	print('test')
        	length=len(x_test[i])
        	while (len(x_test[i]) == len(x_test[i+1])) :
                
                
                
        		        i=i+1
        		        if i > total-2 : 
        		                break

        	size.append(count)
        	i=i+1
        	count=count+1
        	print(i)
        	print(count)
        

        return count


data=pd.read_csv('finaloutput.csv',sep='\t')

data = data.reindex((-data.review.str.len()).argsort()).reset_index(drop=True)

#test_data=pd.read_csv('test.csv')

#data.drop(data.columns[1], axis=1)

#test_data.drop(data.columns[1], axis=1)
x=data.review
y=data.score
#x_test=test_data.data
#y_test=test_data.label
from sklearn.model_selection import train_test_split


x_train, x_valid, y_train, y_valid = train_test_split(x, y,
                                                      test_size=0.1,shuffle=False)


train_steps=step_count(x_train)    
valid_steps=step_count(x_valid)                            
import keras
print('converting')
num_classes=5
y_train=y_train.astype(int)
y_valid=y_valid.astype(int)
y_train = y_train -1
y_valid = y_valid -1
#y = y_test -1


print(y.shape)

#y_testc=y
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
#y_test=keras.utils.to_categorical(y, num_classes)



tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
#sequences = tokenizer.texts_to_sequences(x_train)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


vocabulary_size=len(tokenizer.word_index) + 1
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
embeddings_index = {}
f = open( 'glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = numpy.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = numpy.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
#We load this embedding matrix into an Embedding layer. Note that we set tr
 
numpy.savetxt("foo.csv", embedding_matrix, delimiter=",")










x_train_ids = tokenizer.texts_to_sequences(x_train)
x_valid_ids=tokenizer.texts_to_sequences(x_valid)
'''
x_test_ids=tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train_ids, maxlen=max_words)
x_valid = sequence.pad_sequences(x_valid_ids, maxlen=max_words)
x_test=sequence.pad_sequences(x_test_ids, maxlen=max_words)
'''


model = Sequential()
model.add(Embedding(vocabulary_size, 100, weights=[embedding_matrix], trainable=False))
model.add(LSTM(100))
model.add(Dense(5, activation='softmax'))
"""model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(4, activation='softmax'))"""
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs= 2, batch_size=128, verbose=2)
# Final evaluation of the model
#model = load_model('my_model.h5')



train_steps, train_batches = batch_iter(x_train_ids, y_train,train_steps)
valid_steps, valid_batches = batch_iter(x_valid_ids, y_valid,valid_steps)
model.fit_generator(train_batches, train_steps, epochs=1,validation_data=valid_batches, validation_steps=valid_steps)



print('* Making predictions..')
predictions = model.predict_classes(x_test)
'''
matrix=confusion_matrix(y_testc,predictions)

from sklearn.metrics import precision_recall_fscore_support

print(precision_recall_fscore_support(y_testc, predictions, average='macro',labels='0,1,2,3'))

print(precision_recall_fscore_support(y_testc, predictions, average='micro'))

print(precision_recall_fscore_support(y_testc, predictions, average='weighted'))


numpy.savetxt(
    'test_output.csv',          # file name
    confusion_matrix(y_testc,predictions),  # array to save
    fmt='%.9f',               # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    header= 'probability')   # file header

print 'Predictions saved to /predictions/prediction_03.csv'
print(confusion_matrix(y_testc,predictions))

'''
scores = model.evaluate(x_valid, y_valid, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
