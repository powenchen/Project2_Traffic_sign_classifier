# **Traffic Sign Recognition** 
---

**Goal of the Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./charts/train_data_count.png ""
[image2]: ./charts/test_data_count.png ""
[image3]: ./charts/valid_data_count.png ""
[image4]: ./charts/signs_sample.png ""
[image5]: ./charts/preprocess.png ""
[image6]: ./charts/train_accuracy.png ""
[image7]: ./charts/valid_accuracy.png ""
[image8]: ./charts/test_accuracy.png ""
[image9]: ./GermanSigns/general_caution.jpg ""
[image10]: ./GermanSigns/slippery_road.jpg ""
[image11]: ./GermanSigns/speed_limit_30.jpg ""
[image12]: ./GermanSigns/stop.jpg ""
[image13]: ./GermanSigns/turn_right_ahead.jpg ""
[image14]: ./GermanSigns/yield.jpg ""
[image15]: ./charts/german_result0.png ""
[image16]: ./charts/german_result1.png ""
[image17]: ./charts/german_result2.png ""
[image18]: ./charts/german_result3.png ""
[image19]: ./charts/german_result4.png ""
[image20]: ./charts/german_result5.png ""
[image21]: ./GermanSigns/signs.png ""

### Data Set Summary & Exploration

#### 1. Data Set Summary

The dataset consists 43 classes of traffic sign images.
* Number of classes: 43
* Image Size: All of them are 32x32 pixel RGB images
* Size of training dataset: 34799 images
* Size of testing dataset: 12630 images
* Size of validation dataset: 4410 images

#### 2. An exploratory visualization of the dataset.

Here are histograms of the occurence of different classes of images in 3 datasets 

![alt text][image1]
![alt text][image2]
![alt text][image3]

Here are some samples of images and there class labels

![alt text][image4]


### Design and Test a Model Architecture

#### 1. Image Preprocess

For preprocessing, we first convert the images into grayscale with cv2.cvtColor()

![alt text][image5]

After that we normalize the image to the range of [-1.0,1.0], we can achieve this by using some simple manipulations in python: 
* X_train_gray = (X_train_gray - 128)/ 128


#### 2. My Architecture of NN.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Activation			| RELU											|
| Max pooling	      	| 2x2 stride, outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| Activation   			| RELU        									|
| Max pooling			| 2x2 stride, outputs 5x5x16					|
| Flatten				| flatten the last output from 5x5x16 to 400x1	|
| Fully Connexted Layer	| 400x120, outputs 120x1						|
| Drop out				| drop out probability = 50% when training		|
| Activation			| RELU											|
| Fully Connexted Layer	| 120x84, outputs 84x1							|
| Drop out				| drop out probability = 50% when training		|
| Activation			| RELU											|
| Fully Connexted Layer	| 84x43, outputs 43x1							|
 


#### 3. How I train my model.

To train my model, I'm using a LeNet based neural network architecture with Adam optimizer(provided in tensorflow).

The learning rate here, I'm using a learning rate of 0.001, with batch size of 128 and 30 epochs, the final accuracy is 99.8% in traing_set, 96.3% in validation set and 94.1% in test set.

![alt text][image6]
![alt text][image7]
![alt text][image8]


#### 4. Accuracy.

I'm using a learning rate of 0.001 in my model, with batch size of 128 and 30 epochs, the final accuracy is 99.8% in training set, 96.3% in validation set and 94.1% in test set.

* Accuracy of training dataset: 99.8%
* Accuracy of validation dataset: 96.3%
* Accuracy of test dataset: 94.1%

![alt text][image6]
![alt text][image7]
![alt text][image8]


* What was the first architecture that was tried and why was it chosen?
** I am using this LeNet approach since LeNet was described in Udacity's lesson, they performed pretty good in this problem.


* What were some problems with the initial architecture?
** The accuracy after training wasn't high enough to meet the requirement in this project, so I added one more convolution layer and added dropout mechanism after the fully-connected layers.


* How was the architecture adjusted and why was it adjusted?
** I added one more convolution layer to make the overall architecture a little bit deeper, and I also added dropout mechanism after fully-connected layers to improve the accuracy.

* Which parameters were tuned? How were they adjusted and why?
** I tuned epoch number and batch size for a better accuracy.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
** A convolution layer work well in the problem since the neural network should have some invariance no matter the view angle or position of the traffic sign appear in the training images. And a dropout layer can help the neural network to learn some duplicates of a feature in the image even some part of it is being dropped.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image21]


#### 2. The predictions of our models.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield 	      		| Yield 	   									| 
| Speed limit 30km/h	| Speed limit 20km/h							|
| Caution				| Caution										|
| Stop Sign      		| Stop sign   									| 
| Slippery Road			| Slippery Road      							|
| Turn right ahead 		| Turn right ahead 				 				|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. The only error is a misclassification from 30km/h to 20km/h (and the 2nd suggestion is 30 km/h with 18%).

#### 3. Results on German traffic signs given by our model

The following of the images are the results of the softmax probabilities for each prediction.

![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]


