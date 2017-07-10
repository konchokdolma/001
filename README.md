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

Code shows the example of how we can use the obtained model for a new list of data and outputs the accuracy of the prediction.

Example of an output:

`
acc: 74.22%
`
### titanic.py

In this example we take data from the kaggle competition: https://www.kaggle.com/c/titanic
In the train file we are given the information about passengers in the Titanic. For the input we have their name, age, gender, ticket class, siblings/spouses and parents/children aboard, ticket number, fare, cabin number and port of embarkation.
As an output we have if person survived or not.

First of all code writes the list of data into DataFrame and drops columns with names and ticket numbers. Then in converts columns with strings to integers, i.e. in the column with genders 'male' is '0' and 'female' is '1'.

In the 'train' function we train our model for prediction. We fit the model and then evaluate it for the full data, and then separatly for train and test.

To call the function firstly we drop rows with nan values and then convert all data into integer.

Example of an output:

```
32/712 [>.............................] - ETA: 0s
acc: 83.15%
loss: 5.38%

train accuracy: 83.19% 
train loss: 5.24%

test accuracy: 83.05% 
test loss: 5.66%
```
