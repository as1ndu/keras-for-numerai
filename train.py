#[Import dependencies]
import numpy
import pandas

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#load data
dataframe = pandas.read_csv("data/numerai_training_data.csv")
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:50].astype(float) #X f(eatures) are from the first column and the 50th column
Y = dataset[:,50] # Y (lables) are from the 50th column

#Example Neural Network Architecture
#Define Neural network architecture of 10 Hidden layer with 500 Neurons each
model = Sequential()
model.add(Dense(128, input_dim=50, init='uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu', W_regularizer=l2(0.001) ))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu', W_regularizer=l2(0.001) ))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
print 'Modeled Network'

# Compile model
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
print'* Finished compilling model'

model.fit(X, Y, validation_split=0.3, shuffle=True, nb_epoch=1000, batch_size=1000)
print '* Done training'

# serialize model to JSON
print("* Saving model to disk..")
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model.h5")
print("* Saved model  to disk")

#Compute accuracy scores
print("* Computing accuracy scores..")
score = model.evaluate(X, Y, verbose=0)
print "%s: %.2f%%" % (model.metrics_names[1], score[1]*100)

