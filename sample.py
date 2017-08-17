from classifier import ImageClassifier
from timeit import Timer

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

print "Performance Analysis:"
inception_timer = Timer("res = classifier2.eval(image)", "from __main__ import classifier2,image")
time1 =inception_timer.timeit(20)/20
print "Inception_v3 : {}".format(time1)

mobilenet_timer = Timer("res = classifier1.eval(image)", "from __main__ import classifier1,image")
time2 =mobilenet_timer.timeit(20)/20
print "mobilenet_v1_1.0_224_quantized : {}".format(time2)
print "Ratio: {}".format(time1/time2)
# cv2.imshow("test",image)
# cv2.waitKey(0)