1. Problem Definition - About Project
-------------
The project objective is to implement behavioural cloning – technique whereby, students are required to drive a simulated car, collect data from sensors (camera). 

The simulated car is equipped with three cameras which face the front, are mounted on the left, centre and the right of the simulated car. They also connect the simulator to a scripting back-end so that we can retrieve critical values such as throttle, speed, steering angle, and the images from the simulator during training session. During ‘autonomous’ mode, the scripting back-end can publish a steering angle to the simulator given an image to allow the car to drive.

The collected data during training mode is then passed to the neural network implemented using high-level neural networks library called Keras with Tensorflow at the back-end. The neural network trains on driving data and learns how to drive the car, without any other information. 

Assumption: 
The car is expected to move within driveable portion of the track surface. It need not be aware of other vehicles on the road.

----------

2. Dataset Summary & Exploration
-------------
Collecting data on my own got a bit difficult considering the choices between a keyboard or mouse to control the simulator. However, data provided by Udacity was sufficient and was used to train the network.

The dataset contains,
IMG folder - this folder contains all the image frames of driving.
driving_log.csv - each row in this sheet correlates image with the steering angle, throttle, brake, and speed of your car. Only steering angle data is used for this project.

**Images**
The IMG folder contains about 24,000 JPG images of dimensions 320x160x3. Here are some sample images from the dataset.

![image1](https://cloud.githubusercontent.com/assets/17127066/22473360/ace1c0dc-e7fe-11e6-96c4-f2e31fa65f97.png)

The left and right cameras point straight, along the length of the car. 

![image5](https://cloud.githubusercontent.com/assets/17127066/22473364/acf8ce80-e7fe-11e6-9099-eeba2b374ece.png)

**driving_log.csv – Steering Angle Histogram**

![image3](https://cloud.githubusercontent.com/assets/17127066/22473363/acf164b0-e7fe-11e6-9543-c09d2fcb7c6d.png)

Considering the distribution of steering angles, most of them are close to zero and is slightly unbalanced. There is a bias towards driving straight and a bit for turning left. 

 - Data needs to be augmented for right turns.
 - Considering that in real-time, car tends to move straight for a longer time, decided not to prune data corresponding to zero angles.

**driving_log.csv – Steering Angle Vs Frames (against time)**

![image4](https://cloud.githubusercontent.com/assets/17127066/22473362/acf04846-e7fe-11e6-86cf-059097c99a25.png)

Plotting the steering angle against time, we see that frames seem to be continuous. There is a data structure in a block of frames wherein a frame at time t has relationship with the last t-n frames

 - Since the project involve images, data has spatial information.
 - Since the frames are inter-related, data has temporal information as well
 - So, if we choose to allow the network to learn spatial and temporal features, the network will have two inputs and input data should be prepared accordingly

----------

3. Image pre-processing
-------------

**Analysis**

 - Images from simulator are rectangular. This makes sense since the visibility through car windshield at real-time is stretched. Hence, images should not be resized to square. However, redundant regions (like bonnet at the bottom and sky at the top of the image) can be removed
 - Training is done on a laptop with 4GB RAM, 2GB GPU (nvidia geforce 840M) and Ubuntu 16.04. Considering the hardware constraints, generated images (320x160) are big to process and smaller images are convenient to restrict load on memory and also to minimize the number of parameters.

**Implementation**

 - crop_resize:
After cropping and resizing, the image shape is changed to 120x40x3

 - Re-scaling as the first layer in the model
Re-scaling is required in order to scale the gradients to a manageable level. RGB images are represented by numbers ranging from 0 to 255. This would require large numbers in the weight matrices of a neural network, with correspondingly high gradients. This would make training very difficult using gradient-descent methods. By scaling the input in the range [0-1], this issue can be avoided.
Re-scaling the input in the range [0,1] worked better compared to [-1,1]

Ref: http://www.kdnuggets.com/2016/03/must-know-tips-deep-learning-part-1.html

![image2](https://cloud.githubusercontent.com/assets/17127066/22473361/ace53870-e7fe-11e6-96c1-7df6cca18bf1.png)

 - Adjusting steering angles for right/left images
Images from three cameras [center, left, and right] are provided to address the issue of recovering from being off-center.
A small angle of 0.25 to the left camera and -0.25 to right camera images are added. This is to get left/right cameras closer to the center position. 

Ref: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.xesulytdk

Since simulator in autonomous mode considers just the 'center' images, adjusting steering angles for pre-processing in ‘drive.py’ is not applicable

----------

4. Data Augmentation
-------------
**Analysis**
 
 - From the histogram, data is not balanced and is biased for left turns. Hence data should be augmented for right turns
 - There are a few dark sections in track2 autonomous mode. Since Udacity training data mostly covers bright sections, network needs to learn
 - Track 1 autonomous mose in the simulator is mostly sunny. Most of the images in udacity training data includes roads in sunny scenario. However, track 2 involves sunny and darker sections. Data needs to be augmented to address this scenario.

**Generator**
Training data size is about 350 MB [(120x40x3)*24000]. For the back-propagation to compute gradient, requires twice the memory size that is required for the forward pass.
Training is done on a laptop with 4GB RAM, 2GB GPU (nvidia geforce 840M) and Ubuntu 16.04. Considering the large dataset size and given the system constraint, I had to get images loaded to memory in small batches. 
Keras makes use of that property with its fit_generator function, which expects input presented as generators that infinitely recycle over the underlying data.

*Keras ImageDataGenerator*
Explored Keras ImageDataGenerator to augment images. 

 - I could not find a way to adjust brightness through parameters. 
 - Also, flipping the image horizontally means that corresponding steering values has to be changed which is not possible through ImageDataGenerators

From the above, custom generator was the way to go.

**Implementation**

 - 'flip_image' method 		- helps to augment data for right turns
 - 'adjust_brightness' method 	- helps to augment data for dark patches

![image6](https://cloud.githubusercontent.com/assets/17127066/22473365/ad1eebb0-e7fe-11e6-8170-a5d99525884c.png)

Ref: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.xesulytdk

Ref: https://www.youtube.com/watch?v=bD05uGo_sVI&index=21&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU

Above methods are applied only while training the network. No changes are made in ‘drive.py’

----------

 5. Model Architecture
-------------
**Alternatives assessed**

 1. Nvidia, Comma.ai architectures – Looked at respective documentation to get a handle of these architectures. Have borrowed a few good implementation practices for the custom model.

 2. CNN-RNN Model
Since the dataset include spatial and temporal information, getting the network to learn spatial/temporal representation should make it robust. Below architecture was used to train the network.

![image7](https://cloud.githubusercontent.com/assets/17127066/22474871/447f700c-e803-11e6-8b4e-12e7094eafe0.png)


However, model could not be executed using simulator in autonomous mode. Assume, the simulator is not designed to accept two inputs. 

Ref: https://carnd-forums.udacity.com/questions/19991297/using-left-and-right-cameras-when-driving

**Final Model**

![image8](https://cloud.githubusercontent.com/assets/17127066/22474872/44b880f4-e803-11e6-80e1-3e6e090cb9b6.png)

|Layer (type)                     |Input Shape	       |Output Shape        |Param #     
|:-------			  |----:               |----:	            |:---:
|lambda_1 (Lambda)                |(64, 3, 40, 120)    |(64, 3, 40, 120)    |0           
|convolution2d_1 (32,3x3,Stride=2)|(64, 3, 40, 120)    |(64, 32, 20, 60)    |896         
|maxpooling2d_1 (MaxPooling2D,2x2)|(64, 32, 20, 60)    |(64, 32, 10, 30)    |0           
|convolution2d_2 (32,3x3,Stride=2)|(64, 32, 10, 30)    |(64, 32, 5, 15)     |9248        
|maxpooling2d_2 (MaxPooling2D,2x2)|(64, 32, 5, 15)     |(64, 32, 2, 7)      |0           
|convolution2d_3 (64,3x3,Stride=2)|(64, 32, 2, 7)      |(64, 64, 2, 7)      |18496       
|maxpooling2d_3 (MaxPooling2D,2x2)|(64, 64, 2, 7)      |(64, 64, 1, 3)      |0           
|flatten_1 (Flatten)              |(64, 64, 1, 3)      |(64, 192)           |0           
|activation_1 (Activation)        |(64, 192)           |(64, 192)           |0           
|dense_1 (Dense)                  |(64, 192)           |(64, 500)           |96500       
|dropout_1 (Dropout)              |(64, 500)           |(64, 500)           |0           
|dense_2 (Dense)                  |(64, 500)           |(64, 150)           |75150       
|dropout_2 (Dropout)              |(64, 150)           |(64, 150)           |0           
|dense_3 (Dense)                  |(64, 150)           |(64, 50)            |7550       
|dropout_3 (Dropout)              |(64, 50)            |(64, 50)            |0           
|dense_4 (Dense)                  |(64, 50)            |(64, 1)             |51          
Total params: 207,891
Trainable params: 207,891
Non-trainable params: 0

----------

6. Training
-------------
**Approach**
Training is done on a laptop with 4GB RAM, 2GB GPU (nvidia geforce 840M) and Ubuntu 16.04. Considering the hardware constraints, batch-size of 64 is considered to ensure, we do not run out of GPU memory during the training process. Also, images are loaded to memory in small batches using generators (refer 4. Data Augmentation)

This project is a regression problem. So, it makes sense to use typical regression loss functions such as MSE (mean squared error). 
After training the model for a few epochs, it is tested on Track1. The epoch with the lowest validation loss is not the necessarily the best performing model, and it may even be the model that crashes the car. The final test of how well your model generalises is to let the car run on Track 2 and see how it performs there. 

**Training Parameters**

 - Number of epochs: 3 
 - Optimizer: Adam
 - Training sample size per epoch: 6400
 - Validation sample size per epoch: 1600 
 - The car's performance on the simulator is tested only for, 
  - 640x480 screen resolution 
  - "Fastest" graphics quality
 - No test set used, since the success of the model is evaluated by how well it drives on the road and not by test set loss
 - Generalization:
    - Data augmentation 
    - Batch normalization - did not yield good results 
    - L2 regularisation with weight decay of 0.0001 - did not yield good results

----------

7. Fine-tuning
-------------
**Throttle adjustments**
Throttle level in ‘drive’py’ is initially set to 0.2. This means that the car would have the same throttle at all stretches irespective of whether it going straight and making turns to the left / right. This parameter now varies with speed and does not crash as it turns.

**Last lap in Track2**

< image 8> 
As in the image, this is the last turn before reaching goal in Track2. The car would often go straight and may be, the model is getting confused with the road colour and that of moutain rocks. At epoch 27, the car hit arrow 3 from the right. Had to continue training till epoch 52 before the car could successfully and turn move towards the goal barriers.

**Saving / loading weights from previous runs / epocs**
Since training continued till epoch 52, there were a lot of trial and error checks made in between to check if the model worked. It was not feasible to run training till epoch 52 at one time. As I found that the model made good progress compared to previous runs, weights were saved and this was used as initial weights for subsequent epochs. 
Checkpoints to save weights was effectively used to save weights for all epochs. This made it easier to run multiple epochs at one time and check saved weights using simulator.

----------

8. Further Improvements
-------------
1. Car's performance can further be improved by testing the simulator for all combinations of screen resolutions and graphics quality
2. Additional data under different driving conditions should significantly boost the performance of the model.
3. The car tends to steer from side to side sometimes and can be generalised to make it steady
4. The model could have fewer layers and parameters.

Ref: https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234#.w89x1efbw
