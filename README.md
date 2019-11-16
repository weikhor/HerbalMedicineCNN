# HerbalMedicineCNN

### Mid-Infrared Spectroscopic Fingerprinting and Discrimination of Traditional Herbal Medicine via Multivariate Analysis and Machine Learning 

### Purpose: To discriminate the types of plants using artificial intelligence.

# Multivariate Analysis
![alt text](https://github.com/weikhor/HerbalMedicineCNN/blob/master/math.PNG)

### The project consists of four types of herbal medicine. Each type of plant consists of about 100 datasets. Each dataset consists of many files of different temperature. Each file consists of two columns which are the wavelength of infrared rays and the absorption rate of infrared rays from the plant. Synchronous and asynchronous are equations to find correlation in the reactions of herbal medicine with heat energy. This equations are implemented in parallel computation using python cuda programming in script **gpu.py**. After the execution of parallel code, the 2d correlation image is formed used for analysis. 

# Convolutional Neural Networks (CNN)
![alt text](https://github.com/weikhor/HerbalMedicineCNN/blob/master/vgg.png)

### CNN is a deep Learning algorithm which can take in an image input, assign importance (learnable weights and biases) to various aspects in the image in the traning process and be able to differentiate one from the other in the testing process. In this project, the VGG16 convolutional neural network is implemented using Tensorflow in script <mark>cnn.py</mark>. The sixty percent of datasets are used for training and the remaining datasets are used for testing and validation.   
