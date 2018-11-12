# **Traffic Sign Recognition**

## Kavan Brandon

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load training, validation, and test datasets of german traffic signs images.
* Summarize and explore dataset
* Design and test a model architecture including pre-processing
* Test the model on new images from the web
* Output Top 5 softmax probabilities for each image found on the web
* Summarize the results with a written report

---

[//]: # (Image References)

[image1]: ./write_up_images/random_visualization_image.png "Random Image"
[image2]: ./write_up_images/every_class_image.png "Image from every class"
[image3]: ./write_up_images/label_frequency.png "Label Frequency"
[image4]: ./write_up_images/normalized_output.png "Normalized Output"
[image5]: ./german_traffic_signs/roadwork.png "Roadwork"
[image6]: ./german_traffic_signs/yield.png "Yield"
[image7]: ./german_traffic_signs/right_of_way.png "Right of Way"
[image8]: ./german_traffic_signs/cycling.png "Bicycles Crossing"
[image9]: ./german_traffic_signs/priority_road.png "Priority Road"
[image10]: ./german_traffic_signs/seventy.png "Speed Limit 70"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The first visualization includes an output of a random image from the X_train dataset. This visualization was used in the lesson.

![alt text][image1]

The second visualization includes an image of every class included in the dataset.

![alt text][image2]

The third visualization includes a histogram plotting the frequency of each label in the training set.

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Preprocessing

The first step for preprocessing was shuffling the X-train and y_train sets. This is necessary to ensure random permutations of the training data used by the model.

The second step I took was converting the images to greyscale using np.sum(example_dataset/3, axis=3, keepdims=True) for the training, validation, and test datasets.

The last preprocessing step consisted of normalizing the output of the grayscale dataset. Here is an image comparing the normalized output against the original grayscale image.

![alt text][image4]

#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 gray image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Activation					|				RELU								|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16  				|
| Activation					|				RELU								|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16				|
| Flatten	      	| Input = 5x5x16, Output = 400	|
| Fully connected		| Input = 400, Output = 120.       									|
| Activation					|				RELU								|
| Dropout					|				Probability = .5								|
| Fully connected		| Input = 120, Output = 84      									|
| Activation					|				RELU								|
| Dropout					|				Probability = .5								|
| Fully connected		| Input = 84, Output = 43     									|

#### 3. Model Training

The following parameters were used to train the model:

* Epochs = 60
* Batch Size = 128
* Learning Rate = 0.001
* Optimizer = Adam Optimizer
* mu = 0
* sigma = 0.1

#### 4. Solution Approach

My final model results were:
* Validation set accuracy of .961 and loss of .201
* Test set accuracy of .939

An iterative approach was chosen beginning with the LeNet Architecture:

The LeNet architecture was chosen since it seems to be reliable for smaller convolutional neural networks. The architecture is simple and easier to understand. It also provides reliable results. LeNet works well for images with smaller pixel dimensions. If we were analyzing higher resolution images, it would likely be necessary to use larger and more convolutional layers, in addition to more computing resources.

I generally followed the LeNet modeled used within the CNN lesson on the MNIST dataset. However, I did change the final fully connected layer to output 43 classes instead of 10 in order to match the number of classes provided by the dataset. Initially
the model was run without including a dropout probability which didn't achieve a desirable validation accuracy. I mostly used trial and error principles to achieve a validation accuracy that I felt would perform well on the test set. I experimented using an epoch of 30 and batch size of 256, however, a higher epoch and smaller batch size seemed to perform better. I did notice that around the 25th epoch, the validation accuracy would oscillate between lower and higher percentages, but would generally increase over the remaining epochs.

### Test a Model on New Images

#### 1. Acquiring New Images

Here are six German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10]

Image 1 (Roadwork): There seems to be a tree behind the roadwork sign. This may cause the classifier to have problems detecting the edge of the sign, especially given the images are grayscaled.

Image 2 (Yield): The yield sign has a considerable amount of white clouds in the background.

Image 3 (Right of Way): The right of way sign has a considerable amount of white clouds in the background.

Image 4 (Bicycles Crossing): The bicycles logo could potentially be confused with a similar logo from another sign with a white background and red edges.

Image 5 (Priority Road): The priority road sign has a considerable amount of trees and/or a mound on the bottom end of the sign. Considering the image is grayscaled, it's possible the classifier could have trouble finding the edges of the sign.

Image 6 (Speed Limit 70): The 70 km/hr sign has a large amount of trees on the bottom of the sign. Considering the image is grayscaled, it's possible the classifier could have trouble finding the edges of the sign.

#### 2. Performance on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Roadwork      		| Roadwork  									|
| Yield    			|Yield									|
| Right of Way					| Right of Way												|
| Bicycles Crossing	      		| Beware of ice/snow					 				|
| Priority Road		| Priority Road	  							|
| Speed Limit 70		| Speed Limit 70    							|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 93.9%

#### 3. Model Certainty - Softmax Probabilities

Image 1 (Roadwork):

For the Roadwork image, the model believes this is a Roadwork sign (probability of .99) and is correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99740535        			| Road work 									|
| .002587954     				| Road narrows on the right 										|
| .000006204					| Double curve											|
| .000000420	      			| Bicycles crossing					 				|
| .000000041				    | Dangerous curve to the left      							|

Image 2 (Yield):

For the Yield image, the model believes this is a Yield sign (probability of 1) and is correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Yield 									|
| .000000000     				| Stop										|
| .000000000					| Priority Road										|
| .000000000	      			| Road Work					 				|
| .000000000				    | No Vehicles      							|

Image 3 (Right of Way):

For the Right of Way image, the model believes this is a Right of Way sign (probability of 1) and is correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Right-of-way at the next intersection 									|
| .000000000     				| Beware of ice/snow										|
| .000000000					| Pedestrians										|
| .000000000	      			| Roundabout mandatory					 				|
| .000000000				    | Children crossing      							|

Image 4 (Bicycles Crossing):

For the Bicycles Crossing image, the model believes this is a Beware of ice/snow sign (probability of 0.92) and is not correct. The Bicycles Crossing probability was only the 4th largest. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .916287541         			| Beware of ice/snow  									|
| .054850906     				| Dangerous curve to the right 										|
| .028699157					| Slippery road											|
| .000073645	      			| Bicycles crossing					 				|
| .000043644				    | No passing for vehicles over 3.5 metric tons      							|

Image 5 (Priority Road):

For the Priority Road image, the model believes this is a Priority Road sign (probability of 1) and is correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Priority Road 									|
| .000000000     				| Roundabout mandatory 										|
| .000000000					| Speed limit (100km/h)											|
| .000000000	      			| Yield					 				|
| .000000000				    | No passing      							|

Image 6 (Speed Limit 70):

For the Speed limit (70km/h) image, the model is absolutely sure that this is a Speed limit (70km/h) sign (probability of 0.99) and is correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .999987841         			| Speed limit (70km/h)  									|
| .000012178     				| Speed limit (20km/h) 										|
| .000000001					| General caution											|
| .000000000	      			| Speed limit (30km/h)					 				|
| .000000000				    | Stop      							|
