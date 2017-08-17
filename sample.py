from classifier import ImageClassifier
# wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

import cv2

classifier1 = ImageClassifier("mobilenet_v1_1.0_224_quantized")
classifier2 = ImageClassifier("inception_v3")

image = cv2.imread("/root/sharedfolder/shared/a.png")
print image.shape

res = classifier1.eval(image)
print res
print "inception"
res = classifier2.eval(image)
print res

# cv2.imshow("test",image)
# cv2.waitKey(0)