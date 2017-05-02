# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image4]: writeup_images/CNNvsNN.JPG "Regular Neural Network vs ConvNet"
[image5]: Internet_images/"14-Stop.jpg" "STOP Sign"
[image6]: Internet_images/"25-Road work.jpg" "Road Work"
[image7]: Internet_images/"2-Speed limit (50kmh).jpg" "Speed limit (50kmh)"
[image8]: Internet_images/"2-Speed limit (50kmh) (2).jpg" "Speed limit (50kmh)"
[image9]: Internet_images/"1-Speed limit (30kmh).jpg" "Speed limit (30kmh)"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/nehalsoni23/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

As the data is available in serialized format file, Python's pickle module is being used to unpickle(deserialize) it.

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- The pickled data contains the images of size 32 x 32 pixels

* The size of training set is `34799`
* The size of test set is `12630`
* The shape of a traffic sign image is `(34799, 32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

I have used sklearn module in order to shuffle and split the data.
np.random.seed(80) is used to generate the  same sequence of random numbers.
Shuffeling the training set is very needful in order to avoid overfitting of data.
With the help of sklearn.model_selection module, I have splitted the training set into two parts:
1. Training set is 27839
2. Validation set is 6960

#### 2. Include an exploratory visualization of the dataset.

I have used matplotlib and random packages to visualize the data in notebook

With the use of figure method in matplotlib.pyplot module I created one frame to accomodate randomly choosen 20 files from  training dataset. The Following is few images with its corresponding lables on it.

![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

In our case of predicting signs in images color won't matter. So I am converting the data into grayscale by deviding the matrix by three and then summing the three channels in a marix with choosing the axis 3. The output training set after converting it into grayscale it will have following dimensions: 
(27839, 32, 32, 1) 
The example of color image to grayscale image conversion is shown below:

![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

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
| Fully connected, RELU	| Input = 400, Output = 120	(xW + b)        	|
| Fully connected, RELU	| Input = 120, Output = 84	(xW + b)        	|
| Fully connected, RELU	| Input = 84, Output = 43	(xW + b)        	|


New height and width for next layers are calculated using following formulas:
out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
out_width = ceil(float(in_width - filter_width + 1) / float(strides[1]))

i.e After first convolution layer with (5 x 5 x 1) filter size, height and width for next layer will be calculated as:
out_height = (32 - 5 + 1)/1 = 28	out_width = (32 - 5 + 1)/1 = 28


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

We are using Mini-batch technique which enables us to train data in chunks. Mini-batch is computationally inefficient as we can't caluculate the loss simultaneously. But it can be very useful if computer lacks a memory to store a entire dataset.

I have kept the default value for batch-size(128) and Epochs(30). So my training model will take 27839/128  = 218 iteration for singel epochs. After every epochs, I am evaluating my model with validation dataset. 

For each epochs following operations takes place:
- LeNet model calculates the logits. Cross_entropy is calculated between softmax function applied on logits and one_hot encoded labels. Cross_entropy is used to measure the distance between these two probability vectors.

- The loss is calculated from output of cross_entropy which is basically mean of vector we got as an output of Cross Entropy function.

- I am using AdamOptimizer with a learning rate of 0.001 in order the optimize the weigths and biases.

- The accuracy is calculated using argmax function for logits and one hot vector. Comparing the max value of these two vectors and then casting the resulting boolean vector to float subsequently taking mean of the output matrix will gives us the accuracy.

Above process is repeated for every epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The LeNet architecture is simple and small (in terms of memory footprint), making it perfect for training our model. It can even run on the CPU with good configuration (Of course it will take so much time compare to GPU). 

My final model results are:
* training set accuracy of 0.990
* validation set accuracy of 0.969
* test set accuracy of 0.884

My test accuracy is bit low as I have not added additional data for data augmentation. I believe if adding more complex data on training set would help me increase the accuracy.

If an iterative approach was chosen:
#### What was the first architecture that was tried and why was it chosen?

I started training model with LeNet only as I am still exploring the other well accepted architecture like GoogLeNet, AlexNet etc.

#### Which parameters were tuned? How were they adjusted and why?

Initially I kept learning rate as 0.0001 as smaller learning rate can help improving accuracy. But it didn't worked  well. Then I adjusted it as 0.001 which is quite accepted default setting for learning rate as per my learning. And it worked better than the previous setting.

#### What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Regular Neural Network would not scale well for images as an input. i.e. We have an image of size (28x28x3) so a single fully connected neuron in first hidden layer would need (28*28*3) = 2352 weights. For more bigger images it would the size would lead to unmanagable state. Convolution Neural Network is a best fit when inputs are images because unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth. 

As we can see in below image, the neurons in a layer will only be connected to a small region of the layer before it, instead of all of the neurons in a fully-connected manner.

![alt text][image4]

By http://cs231n.github.io/convolutional-networks/ (Stanford CS class)

If a well known architecture was chosen:
#### What architecture was chosen?
LeNet architecture is best fit for this application.

#### Why did you believe it would be relevant to the traffic sign application?
Since traffic sign classifier is mainly about processing images, ConvNet would be best suitable to process large amount of data and LeNet is based on Convolutional Neural Network.

#### How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
My validation accuracy is 0.968 and training accuracy is 0.990 which is good. Although this can be improved more with the help of adding data augmentation and preprocessing of an image.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image3]

The sixth and seventh image is quite at some angle(not much) but since training data already have images with minor angle, model should be able to predict it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
My lables are:        [14, 25, 2, 2, 1, 34, 3, 18]
The predicted output: [14, 25, 1, 2, 1, 34, 3, 18]

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Road Work    			| Road Work 									|
| Speed limit (50kmh)	| Speed limit (30kmh)						 	|
| Speed limit (50kmh)	| Speed limit (50kmh)					 		|
| Speed limit (30kmh)   | Speed limit (30kmh)      						|
| Turn left ahead       | Turn left ahead      							|
| Speed limit (60kmh)   | Speed limit (60kmh)      						|
| General caution       | General caution      							|

The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. It couldn't predict the first 50kmh speed sign because I believe the the image is at some distance, making the digit very close to each other and add ambiguity for digit '5'. The classifier assumed the digit '5' as '3' because of vague image resolution.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were [14, 13, 39, 25, 34] (indexes = ClassId in CSV file)

![alt text][image5]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop sign   									| 
| 5.31574584e-15     	| Yield 										|
| 1.31443344e-25	    | Keep left										|
| 3.33202251e-29  		| Road work					 					|
| 1.56263420e-30		| Turn left ahead      							|

---

For the second image, top five soft max probabilities were: [25, 39, 10,  1, 11]

![alt text][image6]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.6         			| Road work   									| 
| 0.29    				| Keep left 									|
| 0.04	    			| No passing for vehicles over 3.5 metric tons	|
| 0.02 					| Speed limit (30km/h)					 		|
| 0.001					| Right-of-way at the next intersection      	|

---

For the third image, top five soft max probabilities were: [ 1,  2,  5, 21, 31]

![alt text][image7]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.7         			| Speed limit (30km/h)   						| 
| 0.27    				| Speed limit (50km/h) 							|
| 3.39563225e-19	    | Speed limit (80km/h)							|
| 2.60027457e-20 		| Double curve							 		|
| 8.30955028e-21		| Wild animals crossing					      	|

For this image like I explained before it is misinterpreting digit '5' as '3' which lead to incorrect prediction. The second highest prediction is 50km/h which is answer.

---

For the forth image, top five soft max probabilities were: [ 2,  5,  1, 16,  8]

![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.62         			| Speed limit (50km/h)   						| 
| 0.37    				| Speed limit (80km/h) 							|
| 0.00038			    | Speed limit (30km/h)							|
| 5.97629390e-15 		| Vehicles over 3.5 metric tons prohibited		|
| 1.35115056e-17		| Speed limit (120km/h)					      	|

---

For the fifth image, top five soft max probabilities were: [ 1,  2,  5, 39,  0]

![alt text][image9]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (30km/h)   						| 
| 3.27433912e-21    	| Speed limit (50km/h) 							|
| 9.16801786e-30		| Speed limit (80km/h)							|
| 5.60478021e-33 		| Keep left										|
| 0.00000000e+00		| Speed limit (20km/h)					      	|

Surprisingly the model can identify this image 100% same as stop sign image.

It is noticeable from 3rd, 4th, 5th results that for model there is very small bifurcation between digits '3', '5' and '8'. And we need to take care of this small difference at the time of training the model with proper image resolution and more of such data.
---

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
