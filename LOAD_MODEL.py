from keras.models import model_from_json
import numpy as np

json_file = open('saved.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('saved_weights.h5')
print("Loaded model from disc")

dataset = np.loadtxt("YOUR_FILE.csv", delimiter = ",")

a = dataset.shape[1] - 1
X = dataset[:,0:a]
Y = dataset[:,a]

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
scores = loaded_model.evaluate(X, Y, verbose = 0)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))
