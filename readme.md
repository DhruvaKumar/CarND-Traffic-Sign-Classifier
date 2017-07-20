# **German Traffic Sign Classification** 

[//]: # (Image References)

[image1]: ./output_images/sample_train_data.png "sample train"
[image2]: ./output_images/sample_test_data.png "sample test"
[image3]: ./output_images/dist.png "dist"
[image4]: ./output_images/data_augmentation_og.png "augmentation1"
[image5]: ./output_images/data_augmentation.png "augmentation2"
[image6]: ./output_images/network.png "network"
[image7]: ./output_images/accuracy.png "accuracy"
[image8]: ./output_images/loss.png "loss"
[image9]: ./output_images/new_images.png "new images"
[image10]: ./output_images/top5.png "top5"


Overview
---
The goal of this project is to classify traffic signs. The [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) is used to train and validate a model using CNNs. TensorFlow and Python was used for this purpose.
This project was done as part of Udacity's self-driving nanodegree program. More details of the project can be found [here](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)

The code can be found in [Traffic_Sign_Classifier.ipynb]('./Traffic_Sign_Classifier.ipynb')


### Dependencies

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
* [optional] `pip install tqdm`


Data exploration
---

#### Summary

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Samples of the dataset

Random samples of the dataset:

![alt text][image1]
![alt text][image2]

#### Distribution of the dataset

Number of images per traffic sign class/label:

![alt text][image3]


### Design and Test a Model Architecture

#### Data augmentation

On observing sample images, it can be seen that a lot of images have varying brightness, contrast and occlusions. These effects were randomly added to each image in the training data to simulate these changes. Adding additional data also has the advantage to provide more training data. Deep neural networks have millions of parameters that can be tuned better with more data - especially with data that reflect real world scenario. 

The following perturbations were added to each image to generate a new set of training data:
* Translation (uniform distribution between -5 to 5 pixels)
* Rotation (uniform distribution between -20 to 20 degrees)
* Shear (uniform distribution between -0.1 to 0.1 change along each axis)
* Brightness (uniform distribution between 0.25 to 1.1 scale change of the V channel in the HSV color space)

Affine transformations like translation and rotation have the effect of viewing the signs from different angles. In his excellent blog posts, [Vivek Yadav](https://medium.com/@vivek.yadav) talks about [augmenting data](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3#.sgh2jfdqu) and adding [brightness changes](https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc) that help the model learn better.

Original image

![alt text][image4]

Random perturbations

![alt text][image5]


#### Preprocessing

Based on [Yann LeCun's work](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), I decided to feed in grayscale images instead of RGB images. Adding all 3 dimensions later during hyperparameter tuning lowered the performance. 
The images were also normalized between -1 to 1 to tune the weights easily. For each training image, 4 more jittered images were added increasing the training set from 34799 to 173995.


#### Model

The final model was chosen after trying different architectures. The model is a variation of the LeNet architecture.

![alt text][image6]

We have feedforward network of 3 convolutional layers and 2 fully connected layers. Each convolutional layer is followed by a rectified linear unit activation to add in nonlinearities, max pooling to down sample the images and a dropout regularization so that multiple neurons are forced to learn redundancies in the data which can be averaged later during testing. In addition to preventing overfitting it also acts like an ensemble agent that averages out activations from multiple neurons. 

I started with a LeNet-5 architecture. It worked decently and was giving me a validation accuracy of about 89%. However, the model was overfitting since the training accuracy was high. I added a dropout regularization with a 0.5 probability of retaining neurons after each layer. This improved the validation accuracy, but still wasn't enough. On plotting random images from the dataset, a lot of signs were subjected to either brightness changes, occlusions or affine transformations. I simulated these by adding in jittered images in the training set. By tweaking the parameters further and adding another convolutional layer to learn the augmented data better, I converged upon the network above. 

#### Training

Normalized grayscale images are fed into the network. The output tensor reflects scores for 43 traffic sign classes. A softmax function converts them to probabilities. The loss function is defined as the average cross entropy between these softmax probabilities and one hot encoded labels/traffic sign classes. An Adam optimizer is used to train the network based on this loss function. 

Training parameters:
```
Epochs: 30
Batch size: 128
Learning rate: 0.001 (step decay)
Probability of retaining neurons during dropout: 0.6
```

Accuracy during training:

![alt text][image7]

Cross entropy loss during training:

![alt text][image8]

```
Accuracy:

training set accuracy: 96.8%
validation set accuracy: 94.1%
test set accuracy: 93.1%
```
It can be seen that there's potential for the model to learn the underlying representations better. 


### Test model on new images

Some German traffic sign images were downloaded from the web and classified with the trained model

![alt text][image9]

5 out of 8 images were classified correctly, yielding an accuracy of 62.5%

For each new image, we show the original image, the preprocessed image that is fed into the network and the top 5 softmax probabilities. The green bar shows the correct traffic sign.

![alt text][image10]

We can see that the model has trouble with classifying speed limit 50 and 60 km/hr correctly. It does predict the 50km/hr sign with the second highest probability. Since the sign is viewed from underneath, augmenting with more and aggressive affine transformations can help the model generalize to varied viewpoints. It can also be helpful to have passthrough layers from the lower stages of the network so that it learns multi-scale features as described in [Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). This can help the model discriminate the speed limit signs better.
It also misclassifies the 'Children crossing' sign as 'Right-of-way at the next intersection'. This is possible because of the occlusion caused by the snow. Including occlusions during augmentation can maybe alleviate this. 


### Thoughts

This project was fun and showcased the power of CNNs. It would be interesting to see real-time detection and classification of traffic signs from a camera placed on the dashboard. A variation of [YOLOv2](https://arxiv.org/abs/1612.08242) has the potential to tackle the problem in terms of accuracy and speed.
