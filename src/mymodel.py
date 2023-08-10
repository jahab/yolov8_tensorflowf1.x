import tensorflow as tf
tf.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

import numpy as np 
import logging


# Configure the logger
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Create a logger instance
logger = logging.getLogger(__name__)




def Conv(input_data, filters_shape:tuple, trainable:bool, name=None, downsample=False, activate=True, bn=True, act_fun='leaky_relu'):
    # with tf.variable_scope(name):
    if downsample:
        pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
        paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
        input_data = tf.pad(input_data, paddings, 'CONSTANT')
        strides = (2, 2)
        padding = 'VALID'
    else:
        strides = (1, 1)
        padding = 'SAME'


    conv_layer = tf.layers.Conv2D(filters=filters_shape[-1],
                            kernel_size=(filters_shape[0],filters_shape[1]), 
                            strides=strides, 
                            padding=padding,
                            trainable=trainable,
                            use_bias=True,
                            kernel_initializer = tf.random_normal_initializer(stddev=0.01),
                            name=name)

    if bn:
        bn_layer = tf.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                        beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(),
                                        moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer(),
                                        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                                        trainable=trainable, name=None, renorm=False, renorm_clipping=None,
                                        renorm_momentum=0.99)

    res = conv_layer(input_data)
    res = bn_layer(res)

    if activate:
        res = tf.nn.leaky_relu(res)

    return res

def Bottleneck(input_data,ch_in,ch_out, k=(3,3), shortcut=False):
    
    kernel_shape=(*k,ch_in,ch_out)
    conv1 = Conv(input_data,kernel_shape,True)
    
    logger.info("in bottle nek----{}".format(conv1.shape))
    conv2 = Conv(conv1,kernel_shape,True)
    logger.info("in bottle nek----{}".format(conv2.shape))
    
    if shortcut:
        return conv2
    else:
        logger.info("concat shape {}".format(tf.math.add(conv2,input_data).shape))
        return tf.math.add(conv2,input_data)


def C2F(input_data,ch_in:int,ch_out:int, replicate:int, shortcut:bool,e=0.5):

    c = int(ch_out * e)  # hidden channels
    logger.info("First conv in C2F {} {} {}".format(ch_in, 2 * c, e))
    kernel_shape1 = (1,1,ch_in, 2 * c)
    logger.info("__{} {}".format(input_data.shape, kernel_shape1))
    conv1 = Conv(input_data, kernel_shape1,True)
    
    x = conv1
    splits = tf.split(x,2,3)
    # logger.info("------------",splits, len(splits))
    for i in range(replicate):
        temp = Bottleneck(splits[-1],c,c,shortcut=shortcut)
        logger.info("temp shape {}".format(temp.shape))
        splits.extend([temp])


    concat = tf.concat(splits,3)

    kernel_shape2 = (1,1,(2 + replicate) * c, ch_out)

    conv2 =  Conv(concat,kernel_shape2,True)
    return conv2
    

def SPPF(input_data,ch_in,ch_out, k=(1,1),):
    kernel_shape=(*k,ch_in,ch_out)
    c1 = Conv(input_data,kernel_shape,True,downsample=False)
    maxpool1 = tf.layers.MaxPooling2D(5,1, padding="same")
    m1 = maxpool1(c1)
    m2 = maxpool1(m1)
    m3 = maxpool1(m2)
    concat = tf.concat([c1,m1,m2,m3],3)
    kernel_shape2 = (1,1,ch_out)
    c2 = Conv(concat,kernel_shape2,True)
    return c2


class YOLO():
    def __init__(self,d:float,w:float,r:float, batch_size:int=4,input_shape:tuple=(640,640,3)) -> None:
        self.depth_multiple = d
        self.width_multiple = w
        self.ratio = r
        self.batch_size = batch_size
        self.input_shape = input_shape

    def Network(self, input_data):
        k1 = (3,3,3,64*self.width_multiple)
        c1 = Conv(input_data,k1,True,downsample=True)
        # logger.info(str(c1.shape))
        logger.info("layer 1 execute")
        
        k1 = (3,3,64*self.width_multiple,128*self.width_multiple)
        c2 = Conv(c1,k1,True,downsample=True)
        # logger.info(str(c2.shape))
        logger.info("layer 2 execute")
        c3 = C2F(c2,128,128,2,True)

        return
    




if __name__=="__main__":

    batch_size = 2
    height = 320
    width = 320
    channels = 3
    # input_data = tf.placeholder(tf.float32, shape=(batch_size,height, width,channels))
    x = tf.random.uniform([height*width*channels*batch_size],0,1)
    input_data = tf.reshape(x,[batch_size,width,height,channels])

    k1 = (3,3,3,64)
    c1 = Conv(input_data,k1,True,downsample=True)
    # logger.info(str(c1.shape))
    logger.info("layer 1 execute")
    
    k1 = (3,3,3,128)
    c2 = Conv(c1,k1,True,downsample=True)
    # logger.info(str(c2.shape))
    logger.info("layer 2 execute")
    c3 = C2F(c2,128,128,2,True)
