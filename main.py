#import necessary libraries and packages

import numpy as np
import cv2 as cv
import os.path

# using Zhang et al. model for black and white colourization using Convolutional Neural Networks

# load the black and white images
frame = cv.imread('./images/img2.jpg')
# load the caffemodel and prototxt files
numpy_file = np.load('./models/pts_in_hull.npy')

Caffe_net = cv.dnn.readNetFromCaffe("./models/colorization_deploy_v2.prototxt", "./models/colorization_release_v2.caffemodel")

#creating layers for LAB (caffe model)
numpy_file = numpy_file.transpose().reshape(2, 313, 1, 1)
Caffe_net.getLayer(Caffe_net.getLayerId('class8_ab')).blobs = [numpy_file.astype(np.float32)]
Caffe_net.getLayer(Caffe_net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

# L channel and resize the image
input_width = 224
input_height = 224

rgb_img = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
l_channel = lab_img[:,:,0] 

l_channel_resize = cv.resize(l_channel, (input_width, input_height)) 
l_channel_resize -= 50

# predict a and b channel
Caffe_net.setInput(cv.dnn.blobFromImage(l_channel_resize))
ab_channel = Caffe_net.forward()[0,:,:,:].transpose((1,2,0)) 

(original_height,original_width) = rgb_img.shape[:2] 
ab_channel_us = cv.resize(ab_channel, (original_width, original_height))
lab_output = np.concatenate((l_channel[:,:,np.newaxis],ab_channel_us),axis=2) 
bgr_output = np.clip(cv.cvtColor(lab_output, cv.COLOR_Lab2BGR), 0, 1)
#colourized image output
bgr_output=(bgr_output*255).astype(np.uint8)
cv.imwrite("./result2.png", bgr_output)

# result show
cv.imshow("B/W imgae",frame)
cv.imshow("Colurized image",bgr_output)
cv.waitKey(0)
cv.destroyAllWindows()