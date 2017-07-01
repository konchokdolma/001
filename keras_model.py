from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np

def mod(data, epochs, batch):
    
    dataset = np.loadtxt(data, delimiter=",") #the last column of the file should be the output
    
    a = dataset.shape[1] - 1

    X = dataset[:,0:a]
    Y = dataset[:,a] #output - the last column

    model = Sequential()
    model.add(Dense(25, input_dim=8, activation='elu'))
    model.add(Dense(15, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(5, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))

    #configure the model
    model.compile(loss='logcosh', optimizer='adagrad', metrics=['accuracy']) 

    #training the model 
    model.fit(X, Y, validation_split=.33, epochs = epochs, batch_size = batch, verbose = 0)

    #evaluation of the full list of data
    scores = model.evaluate(X, Y, verbose = 0)
    print("\n(evaluation) %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    #67% of the first columns of the list were taken for a training, the rest - for validation
    b = int(round(.67 * dataset.shape[0] - 1 , 0))
    
    C = dataset[0:b,0:a]
    D = dataset[0:b,a]
    
    scores2 = model.evaluate(C, D, verbose = 0)
    print("\ntrain accuracy: %.2f%% \ntrain loss: %.2f%%" % (scores2[1]*100, scores2[0]*100))
    
    E = dataset[b:,0:a]
    F = dataset[b:,a]
    
    scores3 = model.evaluate(E, F, verbose = 0)
    print("\ntest accuracy: %.2f%% \ntest loss: %.2f%%" % (scores3[1]*100, scores3[0]*100))
    
    g = scores2[1]*100 - scores2[0]*100
    h = scores3[1]*100 - scores3[0]*100
    
    print("\nDifference between accuracy and loss:\nfor train: %.2f%%\nfor test: %.2f%%" % (g , h))
    print("\nRatio: %.2f" % (g/h))

    model_json = model.to_json()
    with open("saved.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("saved_weights.h5")
    print("\nModel has been saved")
    
    return
