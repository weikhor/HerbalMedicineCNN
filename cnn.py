import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import keras
import cv2
from sklearn.model_selection import train_test_split
#import keras


#####################################
place = os.path.abspath(__file__)[::-1][12:][::-1]
item = os.listdir(place + "Edward")

#####################################
store = []
for i in item:
   if(("gpu.py" not in i) and ("nohup.out" not in i) and (("edward.py" not in i))):
     store.append(i)

######################################
item = store

arr = [[[0] for x in range(3601)] for y in range(3601)]
arr = np.array(arr, dtype=object)

x_train = []
y_train = []

######################################
ans = {}
a = 0
for i in item:
    ans[i] = a
    a = a + 1

######################################
I = 0
pixels = []
for i in item:
    File = os.listdir(place + "Edward" + "/" + i)   #files inside plant folder
    for f in File:
       if("ts.png" in os.listdir(place + "Edward" + "/" + i + "/" + f)):
          cnn = place + "Edward" + "/" + i + "/" + f + "/ts.png"

          img = cv2.imread(cnn)
          img = cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
          img = np.sum(img/3, axis=2, keepdims=True)
          Min = np.amin(np.amin(img, axis = 1))
          Max = np.amax(np.amax(img, axis = 1))
          diff = Max - Min
          img = img/diff
          
          pixels = pixels + [img]
          plt.close('all')
          y_train.append(ans[i])
          
          I = I + 1

pixels = np.array(pixels)
x_train_norm = pixels


xTrain, xTest, yTrain, yTest = train_test_split(x_train_norm, y_train, test_size = 0.8, random_state = 123)

x_train_norm = xTrain
y_train = yTrain
####################################################
print(pixels.shape)             

def random_translate(img):
    rows,cols,_ = img.shape
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)
    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst = dst[:,:,np.newaxis]
    return dst

def random_warp(img):
    rows,cols,_ = img.shape
    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06   # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06
    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4
    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    dst = dst[:,:,np.newaxis]
    return dst

def random_scaling(img):   
    rows,cols,_ = img.shape
    # transform limits
    px = np.random.randint(-2,2)
    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])
    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(rows,cols))
    dst = dst[:,:,np.newaxis]
    return dst

def random_brightness(img):
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    return dst
########################################################
input_indices = []
output_indices = []

y_train = np.array(y_train)

for class_n in range(3):
    #print(class_n, ': ', end='')
    class_indices = np.where(y_train == class_n)
    n_samples = len(class_indices[0])
    if n_samples < 500:
        for i in range(500 - n_samples):
            input_indices.append(class_indices[0][i%n_samples])
            output_indices.append(x_train_norm.shape[0])
            new_img = x_train_norm[class_indices[0][i % n_samples]]
            #new_img = random_translate(random_scaling(random_warp(random_brightness(new_img))))
            new_img = random_translate(random_scaling(random_brightness(new_img)))
            x_train_norm = np.concatenate((x_train_norm, [new_img]), axis=0)
            y_train = np.concatenate((y_train, [class_n]), axis=0)
            
            if i % 50 == 0:
                print('|', end='')
            elif i % 10 == 0:
                print('-',end='')
            
    print('')

#print(x_train_norm.shape)
####################################################

#X_train, X_test, y_train, y_test = train_test_split(x_train_norm, y_train, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(x_train_norm, y_train, test_size=0.8, random_state=1)

######################################################
tf.reset_default_graph()
x = tf.placeholder(tf.float32, (None, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 4)


####################################################
def conv(x):
    mu = 0
    sigma = 0.1
    
    #conv1
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 64), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(64))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    conv1 = tf.nn.relu(conv1)
    
    #conv2
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #conv3
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 256), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(256))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
    conv3 = tf.nn.relu(conv3)

    #conv4
    conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(256))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b
    conv4 = tf.nn.relu(conv4)
   
    #conv5
    conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256), mean = mu, stddev = sigma))
    conv5_b = tf.Variable(tf.zeros(256))
    conv5   = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b
    conv5 = tf.nn.relu(conv5)
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #conv6
    conv6_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 512), mean = mu, stddev = sigma))
    conv6_b = tf.Variable(tf.zeros(512))
    conv6   = tf.nn.conv2d(conv5, conv6_W, strides=[1, 1, 1, 1], padding='SAME') + conv6_b
    conv6 = tf.nn.relu(conv6)

    #conv7
    conv7_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv7_b = tf.Variable(tf.zeros(512))
    conv7   = tf.nn.conv2d(conv6, conv7_W, strides=[1, 1, 1, 1], padding='SAME') + conv7_b
    conv7 = tf.nn.relu(conv7)
    
    #conv8
    conv8_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv8_b = tf.Variable(tf.zeros(512))
    conv8  = tf.nn.conv2d(conv7, conv8_W, strides=[1, 1, 1, 1], padding='SAME') + conv8_b
    conv8 = tf.nn.relu(conv8)
    conv8 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    #conv9
    conv9_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv9_b = tf.Variable(tf.zeros(512))
    conv9  = tf.nn.conv2d(conv6, conv9_W, strides=[1, 1, 1, 1], padding='SAME') + conv9_b
    conv9 = tf.nn.relu(conv9)

    #conv10
    conv10_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv10_b = tf.Variable(tf.zeros(512))
    conv10  = tf.nn.conv2d(conv9, conv10_W, strides=[1, 1, 1, 1], padding='SAME') + conv10_b
    conv10 = tf.nn.relu(conv10)
    
    #conv11
    conv11_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv11_b = tf.Variable(tf.zeros(512))
    conv11  = tf.nn.conv2d(conv10, conv11_W, strides=[1, 1, 1, 1], padding='SAME') + conv11_b
    conv11 = tf.nn.relu(conv11)
  
    fc0 = flatten(conv2)
     
    #fc1
    s = int(np.prod(conv11.get_shape()[1:]))
    fc1_W = tf.Variable(tf.truncated_normal(shape=(65536, 4096), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(4096))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1    = tf.nn.relu(fc1)

    #fc2
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(4096, 4096), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(4096))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2    = tf.nn.relu(fc2)
    
    #fc3
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(4096, 4), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(4))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits

print('done')

#########################################################
rate = 0.0001
EPOCHS = 50
BATCH_SIZE = 50
logits = conv(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

##########################################################

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

##########################################################
       
from sklearn.utils import shuffle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("Training.....")
    for i in range(EPOCHS):
        x_train, y_ans = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_ans[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        print("EPOCH {} .....".format(i+1))
        #validation_accuracy = evaluate(X_val, y_val)
        validation_accuracy = evaluate(xTest, yTest)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
  
    #test_accuracy = evaluate(X_test, y_test)
    test_accuracy = evaluate(xTest, yTest)
    print("Testing Accuracy:" , test_accuracy)
    saver.save(sess, './lenet')
    print("Cnn model saved")

##########################################################
#train_accuracy = evaluate(x_train_norm, y_train)
#print("Training Accuracy = {:.3f}".format(train_accuracy))


