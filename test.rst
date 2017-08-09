-----
TRAIN
-----

The absolute minimum use case using an Autonomio dataset is:: 

    from autonomio.commands import *
    %matplotlib inline
    train('text','neg',data('random_tweets'))
    
Using this example and NLTK's sentiment analyzer as an input for the ground truth, Autonomio yields 85% prediction result out of the box with with nothing but:: 

    train('text','neg',data('random_tweets'))
    
There are multiple ways you can input 'x' with single input::

    train('text' ,'neg', data) # a single column where data is string

    train(5, 'neg', data) # a single column by index

    train(['quality_score'], 'neg', data) # a single column by label
    
And few more ways where you input a list for 'x'::

    train([1,5], 'neg', data) # a range of column index

    train(['quality_score', 'reach_score'], 'neg', data) # set of column labels

    train([1,2,4,6,18], 'neg', data) # a list of column index

A slightly more involving example may include changing the number of epochs::

    train('text','neg',data('random_tweets'),epoch=20)
    
For flattening the options are 'mean', 'median', 'none' and IQR. IQR is invoked by inputting a float::

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3)
    
Dropout is one of the most important aspects of neural network::

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3,dropout=.5)
    
You might want to change the number of layers in the network:: 

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3,dropouts=.5,layers=4)

Or change the loss of the model:: 

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3,dropouts=.5,layers=4,loss='kullback_leibler_divergence')

For a complete list of supported losses see [Keras_Losses]_ 

If you want to save the model, be mindful of using .json ending::

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3,save_model='model.json')

Control the neuron size by setting the number of neurons on the input layer:: 

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3,neuron_first=50)

Sometimes changing the batch size can improve the model significantly::

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3,batch_size=15)

By default verbosity from Keras is at mimimum, and you may want the live mode for training:: 

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3,verbose=1)

You can add the shape in the model(the way how layers are distributed)::

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3,verbose=1, shape='brick')

To validate the result to check the test accuracy you may use the validation::

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3,validation=True)

The True for validation puts the half of the data to be trained, the other - tested.

You can also define which part of the data will be validated::

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3,validation=.4)

To be sure about the results which you have got you can use double check::

    train('text','neg',data('random_tweets'),epoch=20,flatten=.3,double_check=True)


TRAIN ARGUMENTS
---------------

Even though it's possible to use Autonomio mostly with few arguments, there are a total 13 arguments that can be used to improving model accuracy::

    def train(X,Y,data,
                dims=300,
                epoch=5,
                flatten='mean',
                dropout=.2,
                layers=3,
                model='train',
                loss='binary_crossentropy',
                save_model=False,
                neuron_first='auto',
                neuron_last=1,
                batch_size=10,
                verbose=0,
                shape='funnel',
                double_check=False,
                validation=False):

+-------------------+-------------------------+-------------------------+
|                   |                         |                         |
| ARGUMENT          | REQUIRED INPUT          | DEFAULT                 |
+===================+=========================+=========================+
| X                 | string, int, float      | NA                      |
+-------------------+-------------------------+-------------------------+
| Y                 | int,float,categorical   | NA                      |
+-------------------+-------------------------+-------------------------+
| data              | data object             | NA                      |
+-------------------+-------------------------+-------------------------+
| epoch             | int                     | 5                       |
+-------------------+-------------------------+-------------------------+
| flatten           | string, float           | 'mean'                  |
+-------------------+-------------------------+-------------------------+
| dropout           | float                   | .2                      |
+-------------------+-------------------------+-------------------------+
| layers            | int (2 through 5        | 3                       |
+-------------------+-------------------------+-------------------------+
| model             | int                     | 'train' (OBSOLETE)      |
+-------------------+-------------------------+-------------------------+
| loss              | string [Keras_Losses]_  | 'binary_crossentropy'   |
+-------------------+-------------------------+-------------------------+
| save_model        | string,                 | False                   |
+-------------------+-------------------------+-------------------------+
| neuron_first      | int,float,categorical   | 300                     |
+-------------------+-------------------------+-------------------------+
| neuron_last       | data object             | 1                       |
+-------------------+-------------------------+-------------------------+
| batch_size        | int                     | 10                      |
+-------------------+-------------------------+-------------------------+
| verbose           | 0,1,2                   | 0                       |
+-------------------+-------------------------+-------------------------+
| shape             | string                  | 'funnel'                |
+-------------------+-------------------------+-------------------------+
| double_check      | True or False           | False                   |
+-------------------+-------------------------+-------------------------+
| validation        | True,False,float(0 to 1)| False                   |
+-------------------+-------------------------+-------------------------+



SHAPES
------


Funnel


Funnel is the shape, which is set by default. It roughly looks like an upside-dowm pyramind, so that the first layer is defined as neuron_max, and the next layers are sligtly decreased compared to previous ones.::


  \          /
   \        /
    \      /
     \    /
      |  |



Long Funnel


Long Funnel shape can be applied by defining shape as 'long_funnel'. First half of the layers have the value of neuron_max, and then they have the shape similar to Funnel shape - decreasing to the last layer.::


 |          |
 |          |
 |          |
  \        /
   \      /
    \    /
     |  |


Rhombus


Rhobmus can be called by definind shape as 'rhombus'. The first layer equals to 1 and the next layers slightly increase till the middle one which equals to the value of neuron_max. Next layers are the previous ones goin in the reversed order.::

     +   +
     /   \
    /     \
   /       \
  /         \
  \         /
   \       /
    \     /
     \   /
     |   |


Diamond


Defining shape as 'diamond' we will obtain the shape of the 'opened rhombus', where everything is similar to the Rhombus shape, but layers start from the larger number instead of 1. ::

    +     + 
   /       \
  /         \
  \         /   
   \       /
    \     /
     \   /
     |   |


Hexagon


Hexagon, which we get by calling 'hexagon' for shape, starts with 1 as the first layer and increases till the neuron_max value. Then some next layers will have maximum value untill it starts to decrease till the last layer. ::

     +  +
    /    \
   /      \
  /        \
 |          |
 |          |
 |          |
  \        /
   \      /
    \    /
     |  |


Brick


All the layers have neuron_max value. Called by shape='brick'. ::


+         +
|         |
|         |
|         |
|         |
 --     --
   |   |


Triangle


This shape, which is called by defining shape as 'triangle' starts with 1 and increases till the last input layer, which is neuron_max. ::


         + +
        /   \
       /     \
      /       \
     /         \
    /           \  
    ----      ----
        |    |

Stairs


You can apply it defining shape as 'stairs'. If number of layers more than four, then each two layers will have the same value, then it decreases.If the number of layers is smaller than four, then the value decreases every single layer. ::

+                    +
|                     |
 ---               ---
    |            |
     ---      ---
        |    |
