# ImageClassifier
How to use:
Currently supports evaluation of Inception_v3 and MobileNets models
1. Download the model files into checkpoints folder. 
    Hierarcy:  checkpoints\architecture_name\frozen_model_file.pb

    Inception_v3:
    mkdir ./checkpoints/inception_v3
    cd ./checkpoints/inception_v3
    wget http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz    
    tar -xzvf inception-2015-12-05.tgz 
    rm inception-2015-12-05.tgz 

    MobileNets:
    mkdir ./checkpoints/mobilenet_v1_1.0_224
    cd ./checkpoints/mobilenet_v1_1.0_224
    wget http://download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz
    tar -xzvf mobilenet_v1_1.0_224_frozen.tgz
    rm mobilenet_v1_1.0_224_frozen.tgz


2. If using model other than inception_v3 or MobileNets, add model info in  create_model_info() in classifier.py.

3. Import and use classifier (example in sample.py)
