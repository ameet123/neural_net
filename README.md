## Neural Network - 

This project performs neural network processing on a variety of sample projects.
All the data is from UCI repository. http://archive.ics.uci.edu/ml/datasets.html 

The projects are spread across 3 main categories,

### 1. Binary Classification

#### Bank: Term Subscription
adding more neurons helps.
higher epoch of 250 helps to improve as well.
smaller batchsize helps
b=25 => accuracy on 1: 21
b=1000 => accuracy on 1:12

### 2. Multi-class classification
#### LandClass
classify land cover based on 29 attributes. However, this data, in the training set, has a lot of noise or errors.
In simplest form, NN gives high error rate.

#### Motor Sensor:
data: http://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis

features from electric drive current. drive has intact and defective components.
Results in 11 different classes with different conditions.
Based on 48 features, predict the class of failures in a motor.
This works out very well due to the extensive dataset as well the inherent relationship in the data.
The accuracy is: 0.9906
AUC score: 0.9948



### 3. Regression