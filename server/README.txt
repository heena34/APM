#LSTM Networks		- Recurrent Neural Networks Problem
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. 


## Environment
* Python 3.6
* numpy 1.13.3
* scipy 0.19.1
* matplotlib 2.0.2
* spyder 3.2.3
* scikit-learn 0.19.0
* h5py 2.7.0 
* Pillow 4.2.1 
* pandas 0.20.3
* Anaconda 3
* TensorFlow 1.3.0
* [Keras 2.1.1](https://keras.io)

## Problem Description
Predicting remaining useful life (or time to failure) of aircraft engines 
https://gallery.cortanaintelligence.com/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2

The network uses simulated aircraft sensor values to predict when an aircraft engine will fail in the future so that maintenance can be planned in advance.

The question to ask is "Given these aircraft engine operation and failure events history, can we predict when an in-service engine will fail?"
Re-formulated the problem statement in the follwoing two different types of machine learning models.
	* Regression models: How many more cycles an in-service engine will last before it fails?
	* Binary classification: Is this engine going to fail within w1 cycles?

## Data
In the **Dataset** directory there are the training, test and ground truth datasets.
The training data consists of **multiple multivariate time series** with "cycle" as the time unit, together with 21 sensor readings for each cycle.
Each time series can be assumed as being generated from a different engine of the same type.
The testing data has the same data schema as the training data.
The only difference is that the data does not indicate when the failure occurs.
Finally, the ground truth data provides the number of remaining working cycles for the engines in the testing data.

## Results of Regression model

|Mean Absolute Error|Coefficient of Determination (R^2)|
|----|----|
|12|0.7965|

## Results of Binary classification 

|Accuracy|Precision|Recall|F-Score|
|----|----|----|----|
|0.97|0.92|1.0|0.96|

## References

- [1] Deep Learning for Predictive Maintenance https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
- [2] Predictive Maintenance: Step 2A of 3, train and evaluate regression models https://gallery.cortanaintelligence.com/Experiment/Predictive-Maintenance-Step-2A-of-3-train-and-evaluate-regression-models-2
- [3] A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository (https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan), NASA Ames Research Center, Moffett Field, CA 
- [4] Understanding LSTM Networks http://colah.github.io/posts/2015-08-Understanding-LSTMs/
         