## 001

### keras_model.py

Code takes the list of data with the last column containing the output(0 or 1), trains it and shows the accuracy and loss for both training and test modes. Also it saves the model and the weights we've obtained which will be used in the next code.

To use the code it is enough to call the function with the line:

`mod("YOUR_FILE.csv", epochs = k, batch = n)`

where the k and n are desired values of number of epochs and the batch size.

Example of an output:
```
(evaluation) acc: 74.74%

train accuracy: 76.07% 
train loss: 7.64%

test accuracy: 72.05% 
test loss: 9.34%

Difference between accuracy and loss:
for train: 68.43%
for test: 62.71%

Ratio: 1.09

Model has been saved
```

### load_model.py

Code shows the example of how we can use the obtained model for a new list of data and show the accuracy of the prediction.

Example of an output:

`
acc: 74.22%
`
