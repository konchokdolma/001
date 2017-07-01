## 001

### KERAS_MODEL.py

Code takes the list of data with the last column containing the output(0 or 1), trains it and shows the accuracy and loss for both training and test modes. Also it saves the model and the weights we've obtained which will be used in the next code.

To use the code it is enough to call the function with the line:

`mod("YOUR_FILE.csv", epochs = k, batch = n)`

where the k and n are desired values of number of epochs and the batch size.

### LOAD_MODEL.py

Code shows the example of how we can use the obtained model for a new list of data and show the accuracy of the prediction.
