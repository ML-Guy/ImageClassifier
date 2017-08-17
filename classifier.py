
"""Simple image classification with Mobilenet/Inception model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

def create_model_info(architecture, model_file_name=None, label_file_name=None):
    """Given the name of a model architecture, returns information about it.

    There are different base image recognition pretrained models that can be
    retrained using transfer learning, and this function translates from the name
    of a model to the attributes that are needed to download and train with it.

    Args:
        architecture: Name of a model architecture.

    Returns:
        Dictionary of information about the model, or None if the name isn't
        recognized

    Raises:
        ValueError: If architecture name is unknown.
    """
    architecture = architecture.lower()
    if architecture == 'inception_v3':
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        if not model_file_name:
            model_file_name = os.path.join("checkpoints",'inception_v3','classify_image_graph_def.pb')
        if not label_file_name:
            label_file_name = os.path.join("checkpoints",'inception_v3','imagenet_2012_challenge_label_map_proto.pbtxt')
        input_mean = 128
        input_std = 128
    elif architecture.startswith('mobilenet_'):
        size = int(architecture.split("_")[3])
        is_quantized = bool(len(architecture.split("_")) == 5 and architecture.split("_")[4] == "quantized")
        dirname = "_".join(architecture.split("_")[:4])
        if not model_file_name:
            if is_quantized:
                model_base_name = 'quantized_graph.pb'
            else:
                model_base_name = 'frozen_graph.pb'
            model_file_name = os.path.join("checkpoints",dirname,model_base_name)

        if not label_file_name:
            label_file_name = os.path.join("checkpoints",'inception_v3','imagenet_2012_challenge_label_map_proto.pbtxt')

        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        bottleneck_tensor_size = 1001
        input_width = size
        input_height = size
        input_depth = 3
        resized_input_tensor_name = 'input:0'
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
            'output_layer': bottleneck_tensor_name,
            'output_layer_size': bottleneck_tensor_size,
            'label_file_name': label_file_name,
            'input_width': input_width,
            'input_height': input_height,
            'input_depth': input_depth,
            'input_layer': resized_input_tensor_name,
            'model_file_name': model_file_name,
            'input_mean': input_mean,
            'input_std': input_std,
    }

def get_model_labels(label_file_name,output_layer_size):
    #TODO: To be completed!!
    return range(output_layer_size)

def load_graph(filename):
    """Unpersists graph from file as default graph."""
    with tf.Graph().as_default() as graph:
        with tf.gfile.FastGFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        return graph

def adapt_image_to_model(graph, input_width, input_height, input_depth, input_mean,
                                            input_std):
    """Adds operations that perform image resizing and rescaling to the graph..

    Args:
        input_width: Desired width of the image fed into the recognizer graph.
        input_height: Desired width of the image fed into the recognizer graph.
        input_depth: Desired channels of the image fed into the recognizer graph.
        input_mean: Pixel value that should be zero in the image for the graph.
        input_std: How much to divide the pixel values by before recognition.

    Returns:
        Tensors for the node to feed image data into, and the output of the
            preprocessing steps.
    """
    with graph.as_default():
        decoded_image = tf.placeholder(dtype=tf.float32, shape=[None, None, 3], 
                                        name="Image_Placeholder")
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        resize_shape = tf.stack([input_height, input_width])
        resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
        resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
        offset_image = tf.subtract(resized_image, input_mean)
        mul_image = tf.multiply(offset_image, 1.0 / input_std)
    
    return decoded_image, mul_image

class ImageClassifier:
    """Given the name of a model architecture, returns classifier from it.

    There are different base image recognition pretrained models that can be
    retrained using transfer learning, and this builds classifier model based
    on name and information present in create_model_info function.  

    Typical usage:
    classifier = ImageClassifier("mobilenet_v1_1.0_224_quantized")
    classification_results = classifier.eval(image_data_numpyarray)

    Function:
    ImageClassifier()       : Initialize the classifier
    ImageClassifier.eval()  : Run classification on image and return results
    ImageClassifier.close() : Stops classifier session. Clears memory taken by session object
    ImageClassifier.open()  : Reallocate session for classifier. Used if ImageClassifier.close()
                              has been called to free-up session memory.

    """
    def __init__(self,architecture,model_file_name=None):
        self.architecture = architecture
        self.info = create_model_info(architecture,model_file_name)
        self.labels = get_model_labels(self.info["label_file_name"], self.info["output_layer_size"])
        self.graph = load_graph(self.info["model_file_name"])
        
        (self.resize_ip_tensor,self.resize_op_tensor) = adapt_image_to_model(self.graph, self.info["input_width"], self.info["input_height"],
                                                                            self.info["input_depth"], self.info["input_mean"],
                                                                            self.info["input_std"])

        self.input_layer = self.graph.get_tensor_by_name(self.info["input_layer"])
        self.output_layer = self.graph.get_tensor_by_name(self.info["output_layer"])


        self.session = tf.Session(graph=self.graph)


    def eval(self,image_data,num_top_predictions=5, adjustImage=True):
        result = []
        if self.session:
            if adjustImage:
                resized_input_values = self.session.run(self.resize_op_tensor, feed_dict={self.resize_ip_tensor: image_data})
                predictions, = self.session.run(self.output_layer, feed_dict={self.input_layer: resized_input_values})
            else:
                predictions, = self.session.run(self.output_layer, feed_dict={self.input_layer: image_data})

            # Sort to show labels in order of confidence
            top_k = predictions.argsort()[-num_top_predictions:][::-1]

            result = [{"id": node_id, "label":self.labels[node_id], "score": predictions[node_id]} for node_id in top_k]
        else:
            raise RuntimeError('Classifier session closed. Open it again.')

        return result

    def close(self):
        self.session.close()
        self.session = None

    def open(self):
        self.session = tf.Session(graph=self.graph)



