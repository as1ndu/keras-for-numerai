from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import pandas

#load data
dataframe = pandas.read_csv("data/numerai_training_data.csv")
dataset = dataframe.values
X = dataset[:,:50].astype(float)


# load json and create model
json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("models/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)

print('* Making predictions..')
predictions = loaded_model.predict(X)

numpy.savetxt(
    'predictions/predictions.csv',          # file name
    predictions,  # array to save
    fmt='%.9f',               # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    header= 'probability')   # file header

print 'Predictions saved to /predictions/prediction_01.csv'
