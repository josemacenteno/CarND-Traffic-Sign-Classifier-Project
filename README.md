**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)



[image_2_25]: ./writup_images/2_25_difficult_images.png "Images Hard to clasiffy"
[image_hist_y]: ./writup_images/hist_y_train.png "Train set class histogram"
[image_new_images]: ./writup_images/new_images_from_google_maps.png "Resized new imaged (32x32)"
[image_new_pp]: ./writup_images/new_images_post_processed.png "New images post-processed"
[image_train_pp]: ./writup_images/post_processed_train_images.jpg "Post-processed image"
[image_random_train]: ./writup_images/random_train_images.jpg "Visualization"
[image_new_1]: ./new_images/1.jpg "New Image 1"
[image_new_17]: ./new_images/17.jpg "New Image 17"
[image_new_18]: ./new_images/18.jpg "New Image 18"
[image_new_28]: ./new_images/28.jpg "New Image 28"
[image_new_38]: ./new_images/38.jpg "New Image 38"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/josemacenteno/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

There is an html version [here](https://github.com/josemacenteno/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I converted the input images to np, which will later be useful for image preprocessing. 

I used numpy.shape and python's built in function "len" to calculate basic statistics about the german traffic sign data set used. In particular here are the answers to the questions proposed in the Jupiter notebook:

* The size of training set is 34,799 images
* The size of test set is 12,630 images
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in code cells 4-7 code cells of the IPython notebook. It is a random showing of 30 traffic signs from the training data set, and a histogram.

![alt text][image_random_train]

The histogram counts how many images belong to each of the 43 classes, to give an idea of how balanced is the data set.
![alt text][image_train_histogram]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in code cells 8-9  of the IPython notebook.

The first step was to do to do a min-max rescaling to normalize the data. The min_max_scale function implements the normalization. Before I added more pre-processing techniques, I tried some trial trainings on basic models...

The LeNet model from the Udacity lecture lab was first. It gave me a validation accuracy of 85%. I played a little bit with the batch size. I notice d my PC can handle up to 1024 batches, but the memory is almost 100% used, so I kept the batches at 256 to make sure I always have enough memor. even if the model grows I thuink my PC can handle batches of 256.

I also played with the learning rate. The best validation accuracy correspondes to a learning rate of 0.01. The validation accuracy was 0.90

After tuning the initial hyper parameters I tried with grayscale images.This pre-processing technique was suggested by the instructions of the project itself.No improvement or penalty was measured in terms of accuracy. I left the gray-scale step since this makes the model smaller.

Before I started adding more steps I tried adding a tensof flow "visualization" for the wrong prediction. Some extra nodes in the graph extract the predictions that were misclassified:
```
incorrect_prediction = tf.logical_not(correct_prediction)
wrong_tags = tf.boolean_mask(y, tf.reshape(incorrect_prediction, shape = [-1]))
```

This revealed the images of classes 2 and 25 where difficult. I looked at the validation data set.

![alt text][image_2_25]

There was nothing special about the patterns for this signs, so it was susprising to see everytime a tag was incorrectly predicted it corresponded to one of these tags. I thought this could be explained by either having very difficult images in the validation set for these classes, or having an overfit training on them. The overfit explanation is appealing since those are two of the most numerous classes in the training data set. I will described how I deal with overfitting later.

It looked to me like the difficult to classify images were too dark. To improve the contrast on most images I added a pre-processing stage to do histogram equalization. This makes use of the full gray scale color spectrum, which increases the contrast on images. This is a very common step for image processing applications. This is the last step of pre-processing added. Here is a visualization of the trainning data after pre-processing:

![alt_text][image_new_pp]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The trainning set was divided in train and valid tags. As described in the pre-processing visualization some of the images in the validation set were always problematic for the LeNet model defined in code cell 10. I wanted to include the images in the validation set as part of the training data set. I opted to merge them in cell 3. 

Since I didn keep a eparate static validation set, I opted to do cross validation on my model.

Cell 11 handles the batching and cross validation splitting. I randomly split the training data into a training set and validation set. Here is the code used:
```
X_train_pp, y_train_pp = shuffle(X_train_pp, y_train_pp)
X_xval_train, X_xval_valid, y_xval_train, y_xval_valid = train_test_split(X_train_pp, y_train_pp, test_size=0.20, random_state=i)
num_examples = len(X_xval_train)
for offset in range(0, num_examples, BATCH_SIZE):
       end = offset + BATCH_SIZE
       batch_x, batch_y = X_xval_train[offset:end], y_xval_train[offset:end]
       sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep: dropout_keep_prob})
           
 validation_accuracy = evaluate(X_xval_valid, y_xval_valid)      
```
Now, there was already a concern for over fitting given that the trainning data set is not balanced. I am making this a bit worse by doing cross-validation and trainning the model on validation images during other epochs. To counter act this I added a dropout layer after each activation function of the LeNet model. I tried several values for dropout but it was clear to me that keep probabilities under 0.75 where to aggressive. Since I really wanted to get closer to 0.5 I made the model larger, able to handle features on more neurons during the conv net. I made the Convolutional layer almost twice as large and settled for a dropout keep probability of 0.6

Using drop out keep rate of 0.75 I got cross-validation accuracy numbers aroung 0.93. After making the convolutional filter depth twice as large and lowering the drop-out keep rate to 0.6 I got up to 0.98, with just a few miss-predicitons. I am happy with this values.

I tested on the images that were provided as a test set and see 0.92 validation accuracy, which is 7% higher than the original numbers in the static validation set. This means that the techniques used are effective, but we can still find room for improvement.

Data augmentation to balance the training set can help to avoid overfitting beyond what drop out can do. This might translate into a test validation accuracy closer to the 0.98 we see in the cross-validation accueracy calculations.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scale images   							| 
| Convolution 5x5  | 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU + Dropout		| 												|
| Max pooling	     | 2x2 stride,  outputs 14x14x16 				| 
| Convolution 5x5  | 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU + Dropout		| 												|
| Max pooling	     | 2x2 stride,  outputs 5x5x32 				|
| Fully connected		| Flat input of 800 features and 120 outputs			|
| RELU + Dropout		| 												|
| Fully connected		| Flat input of 120 features and 84 outputs			|
| RELU + Dropout		| 												|
| Fully connected		| Flat input of 84 features and 43 outputs			|
| Softmax	+ reduce_mean			| used for trainning operation (Error calculation).	|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on google maps (downtown Berlin):

![alt text][image_new_1] ![alt text][image_new_17] ![alt text][image_new_18] 
![alt text][image_new_28] ![alt text][image_new_38]

I got all the images from Google Maps. I think the contrast is pretty good on all of them. 

The first image may be diffucult. The 3 is very clearly marked and the contrast is better than most training images, but the validation set reported problems to classify speed limit images in general.

The fourth image might me difficult to classify since the figure in the middle has a lot of details, and there are many similar triangular images. It seems like 32x32 pixels is not enough to describe the figure in detail.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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
