#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: writeup_images/training-data.JPG "Visualization"
[image2]: writeup_images/grayscale_conversion.JPG "Grayscaling"
[image3]: writeup_images/internet_images.JPG "Internet Images"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

As the data is available in serialized format file, Python's pickle module is being used to unpickle(deserialize) it.

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- The pickled data contains the images of size 32 x 32 pixels

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (34799, 32, 32, 3)
* The number of unique classes/labels in the data set is 43

I have used sklearn module in order to shuffle and split the data.
np.random.seed(80) is used to generate the  same sequence of random numbers.
Shuffeling the training set is very needful in order to avoid overfitting of data.
With the help of sklearn.model_selection module, I have splitted the training set into two parts:
1. Training set is 27839
2. Validation set is 6960

####2. Include an exploratory visualization of the dataset.

I have used matplotlib and random packages to visualize the data in notebook

With the use of figure method in matplotlib.pyplot module I created one frame to accomodate randomly choosen 20 files from  training dataset. The Following is few images with its corresponding lables on it.

![alt text][image1]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

In our case of predicting signs in images color won't matter. So I am converting the data into grayscale by deviding the matrix by three and then summing the three channels in a marix with choosing the axis 3. The output training set after converting it into grayscale it will have following dimensions: 
(27839, 32, 32, 1) 
The example of color image to grayscale image conversion is shown below:

![alt text][image2]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I have used LeNet model architecture for training my model which consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5    	| 1x1 stride, VALID padding, Outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 2x2 filter, Outputs 14x14x6 		|
| Convolution 5x5	    | 1x1 stride, VALID padding, Output 10x10x16    |
| RELU					|         										|
| Max pooling	      	| 2x2 stride, 2x2 filter, Outputs 5x5x16 		|
| Flatter				| 5x5x16 = 400        							|
| Fully connected, RELU	| Input = 400. Output = 120	(xW + b)        	|
| Fully connected, RELU	| Input = 120. Output = 84	(xW + b)        	|
| Fully connected, RELU	| Input = 84. Output = 43	(xW + b)        	|


New height and width for next layers are calculated using following formulas:
out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
out_width = ceil(float(in_width - filter_width + 1) / float(strides[1]))

i.e After first convolution layer with (5 x 5 x 1) filter size, height and width for next layer will be calculated as:
out_height = (32 - 5 + 1)/1 = 28	out_width = (32 - 5 + 1)/1 = 28


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

We are using Mini-batch technique which enables us to train data in chunks. Mini-batch is computationally inefficient as we can't caluculate the loss simultaneously. But it can be very useful if computer lacks a memory to store a entire dataset.

I have kept the default value for batch-size(128) and Epochs(30). So my training model will take 27839/128  = 218 iteration for singel epochs. After every epochs, I am evaluating my model with validation dataset. 

For each epochs following operations takes place:
- LeNet model calculates the logits. Cross_entropy is calculated between softmax function applied on logits and one_hot encoded labels. Cross_entropy is used to measure the distance between these two probability vectors.

- The loss is calculated from output of cross_entropy which is basically mean of vector we got as an output of Cross Entropy function.

- I am using AdamOptimizer with a learning rate of 0.001 in order the optimize the weigths and biases.

- The accuracy is calculated using argmax function for logits and one hot vector. Comparing the max value of these two vectors and then casting the resulting boolean vector to float subsequently taking mean of the output matrix will gives us the accuracy.

Above process is repeated for every epochs.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.990
* validation set accuracy of 0.969
* test set accuracy of 0.884

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image3]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


