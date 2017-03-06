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

![alt text][image_hist_y]

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

![alt_text][image_train_pp]


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
| RELU + Dropout		| Keep probability of 0.6						|
| Max pooling	     | 2x2 stride,  outputs 14x14x16 				| 
| Convolution 5x5  | 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU + Dropout		| Keep probability of 0.6				|
| Max pooling	     | 2x2 stride,  outputs 5x5x32 				|
| Fully connected		| Flat input of 800 features and 120 outputs			|
| RELU + Dropout		| Keep probability of 0.6				|
| Fully connected		| Flat input of 120 features and 84 outputs			|
| RELU + Dropout		| Keep probability of 0.6					|
| Fully connected		| Flat input of 84 features and 43 outputs			|
| Softmax	+ reduce_mean			| Used for trainning operation (Error calculation).	|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eleventh cell of the ipython notebook. 

To train the model, I reused a lot of the code from the LeNet lab. It uses a cross_entropy_softmax function with a mean_reduce operation as the error calculation. It uses an adam_optimizer fro tranning. THe validation accuracy is calcualted by finding the largest weight in the logits and taking it as the prediction. We then use the label to compare the error:
```
logits = LeNet(x, keep)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
Each epoc splits the data, trains on every image in the tranning set and then evaluates the model using the cross-validation set defined at the beggining of the epoch.

After about 7 epochs I didn't see much progress, so I lowered the learning rate to 0.05, and increased the number of epochs to 20. The saturation is reached aroung epoch 11 though.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eleventh cell of the Ipython notebook.

My final model results were:
* Cross-validation set accuracy of 0.969 
* test set accuracy of 0.919

Here is a summary of the techniques used:


If a well known architecture was chosen:
* The LeNet architecture was used as a starting point
* After adding aggresive dropout validation accuracy could fall up to 0.40
* Not so aggresive dropout keep probability of 0.6 was chosen, but the depth of the convolution filters was doubled.
* Learning rate and epoch were tuned after every major pre-processing step or architecture change.
* The LeNet based architecture is great for images. The convolutional layers at the input extract the patterns that define the different traffic signs regardless of the position or roation of the sign in the 32x32 image input. The fully connected layers also do a good job at classifying the features extracted on the convolutional steps.
* The cross validation reached 0.98 accuracy, and didn make any prediction mistake on some epochs. This proves the architecture and changes made are good for the task at hand.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on google maps (downtown Berlin):

![alt text][image_new_1] ![alt text][image_new_17] ![alt text][image_new_18] 
![alt text][image_new_28] ![alt text][image_new_38]

I got all the images from Google Maps. I think the contrast is pretty good on all of them. I basically walked around Berlin downtown on street view and captured the first 5 signs I found that had a corresponding class in the dataset analyzed.

From the new images The first one may be diffucult. The 3 is very clearly marked and the contrast is better than most training images, but the testing set reported problems to classify some speed limit images.

The fourth image might me difficult to classify since the figure in the middle has a lot of details, and there are many similar triangular images. It seems like 32x32 pixels is not enough to describe the figure in enough detail.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. I

Here is a visualization of the new images after resize and pre-processing:
![alt_text][image_new_pp]

The code for making predictions on my final model is located in the fiftinth cell of the Ipython notebook.

Here are the results of the prediction:

| Image | Prediction | 
|:---------------------:|:---------------------:|
| No entry | No entry | 
| Children crossing | Children crossing | 
| General caution | General caution | 
| Speed limit (30km/h) | Speed limit (30km/h) |  
| Keep right | Keep right | 


The model was able to correctly guess all 5 of the 5 traffic signs, which gives an accuracy of 100%. This is an even better result than the test set. Again this is probably due to the good quality of the images.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

For the first image, the model is absolutely sure that this is a No entry sign, which is correct.

| Probability | Prediction | 
|:---------------------:|:---------------------:|
| 1.000 | No entry |
| 0.000 | Stop |
| 0.000 | Roundabout mandatory |
| 0.000 | Keep right |
| 0.000 | Turn left ahead |



For the second image, the model is very uncertain. It gives a 21.6% chance of the image being a children crossing, closely followed by two predictions of images that are also triangles with black figures in the middle. The prediction is correct, but the certainty is not high.

| Probability | Prediction | 
|:---------------------:|:---------------------:|
| 0.216 | Children crossing |
| 0.163 | Beware of ice/snow |
| 0.113 | Pedestrians |
| 0.093 | Dangerous curve to the right |
| 0.071 | Bicycles crossing |


Images 3 to 5 are very confident on their prediction. Looks like the speed limit sign was easy, contrary to my expectation. I really think this might be related to how good the images are in Google street view.

| Probability | Prediction | 
|:---------------------:|:---------------------:|
| 0.981 | General caution |
| 0.011 | Pedestrians |
| 0.007 | Traffic signals |
| 0.000 | Dangerous curve to the right |
| 0.000 | Road narrows on the right |


| Probability | Prediction | 
|:---------------------:|:---------------------:|
| 1.000 | Speed limit (30km/h) |
| 0.000 | Speed limit (20km/h) |
| 0.000 | Speed limit (50km/h) |
| 0.000 | Speed limit (70km/h) |
| 0.000 | Speed limit (80km/h) |


| Probability | Prediction | 
|:---------------------:|:---------------------:|
| 1.000 | Keep right |
| 0.000 | Turn left ahead |
| 0.000 | No vehicles |
| 0.000 | Stop |
| 0.000 | Yield |

