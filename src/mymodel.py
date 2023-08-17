import tensorflow as tf
tf.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

import numpy as np 
import logging
import traceback

# Configure the logger
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Create a logger instance
logger = logging.getLogger(__name__)




def Conv(input_data, filters_shape:tuple, trainable:bool, name=None, downsample:bool=False, activate:bool=True, bn:bool=True, act_fun='leaky_relu'):
    
    """
    params: 
    input_data: input_tensor
    filters_shape: 1D tuple of length 4, [kernerl_h,kernel_w,channel_in, channel_out]
    trainable: true/false
    """
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
    
    # logger.info("in bottle nek----{}".format(conv1.shape))
    conv2 = Conv(conv1,kernel_shape,True)
    # logger.info("in bottle nek----{}".format(conv2.shape))
    
    if shortcut:
        return conv2
    else:
        # logger.info("concat shape {}".format(tf.math.add(conv2,input_data).shape))
        return tf.math.add(conv2,input_data)


def C2F(input_data,ch_in:int,ch_out:int, replicate:int, shortcut:bool=False,e=0.5):

    c = int(ch_out * e)  # hidden channels
    # logger.info("First conv in C2F {} {} {}".format(ch_in, 2 * c, e))
    kernel_shape1 = (1,1,ch_in, 2 * c)
    # logger.info("__{} {}".format(input_data.shape, kernel_shape1))
    conv1 = Conv(input_data, kernel_shape1,True)
    
    x = conv1
    splits = tf.split(x,2,3)
    # logger.info("------------",splits, len(splits))
    for i in range(replicate):
        temp = Bottleneck(splits[-1],c,c,shortcut=shortcut)
        # logger.info("temp shape {}".format(temp.shape))
        splits.extend([temp])


    concat = tf.concat(splits,3)

    kernel_shape2 = (1,1,(2 + replicate) * c, ch_out)

    conv2 =  Conv(concat,kernel_shape2,True)
    return conv2
    

def SPPF(input_data,ch_in,ch_out, k=(1,1)):
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


class YOLOV8():
    def __init__(self,input_data, d:float=0.33,w:float=0.50,r:float=2.0, batch_size:int=4,input_shape:tuple=(640,640,3)) -> None:
        self.depth_multiple = d
        self.width_multiple = w
        self.ratio = r
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.input_data = input_data
    
    def Backbone(self):
        try:
            k1 = (3,3,3,int(64*self.width_multiple))
            c0 = Conv(self.input_data,k1,True,downsample=True)
            logger.info("layer 0 execute {}".format(c0.shape))
            
            k1 = (3,3,int(64*self.width_multiple),int(128*self.width_multiple))
            c1 = Conv(c0,k1,True,downsample=True)
            logger.info("layer 1 execute {}".format(c1.shape))
            
            c2 = C2F(c1,int(128*self.width_multiple),int(128*self.width_multiple),int(3*self.depth_multiple),True)
            logger.info("layer 2 execute {}".format(c2.shape))
            
            k1 = (3,3,int(128*self.width_multiple),int(256*self.width_multiple))
            c3 = Conv(c2,k1,True,downsample=True)
            logger.info("layer 3 execute {}".format(c3.shape))
            
            c4 = C2F(c3,int(256*self.width_multiple),int(256*self.width_multiple),int(6*self.depth_multiple),True)
            logger.info("Layer 4 execute {}".format(c4.shape))

            k1 = (3,3,int(256*self.width_multiple),int(512*self.width_multiple))
            c5 = Conv(c4,k1,True,downsample=True)
            logger.info("layer 5 execute {}".format(c5.shape))

            c6 = C2F(c5,int(512*self.width_multiple),int(512*self.width_multiple),int(6*self.depth_multiple),True)
            logger.info("Layer 6 execute {}".format(c6.shape))

            k1 = (3,3,int(512*self.width_multiple),int(512*self.width_multiple*self.ratio))
            c7 = Conv(c6,k1,True,downsample=True)
            logger.info("layer 7 execute {}".format(c7.shape))

            c8 = C2F(c7,int(512*self.width_multiple*self.ratio),int(512*self.width_multiple*self.ratio),int(3*self.depth_multiple),True)
            logger.info("Layer 8 execute{}".format(c8.shape))

            c9 = SPPF(c8,int(512*self.width_multiple*self.ratio),int(512*self.width_multiple*self.ratio))
            logger.info("Layer 9 execute {}".format(c9.shape))
        except:
            logger.error(traceback.print_exc)
            return None
        return c4,c6,c9

    def Head(self, c4,c6,c9):
        c10 = tf.image.resize_images(c9, (40,40), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        logger.info("Layer 10 execute {}".format(c10.shape))
        
        c11 = tf.concat([c10,c6],3)
        logger.info("Layer 11 execute {}".format(c11.shape))
        
        shape = c11.shape        
        c12 = C2F(c11,shape[-1].value, int(512*self.width_multiple),int(3*self.depth_multiple),shortcut=False)
        logger.info("Layer 12 execute {}".format(c12.shape))

        c13 = tf.image.resize_images(c12, (80,80), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        logger.info("Layer 13 execute {}".format(c13.shape))
        
        c14 = tf.concat([c13,c4],3)
        logger.info("Layer 14 execute {}".format(c14.shape))

        shape = c14.shape
        c15 = C2F(c14,shape[-1].value, int(256*self.width_multiple) ,int(3*self.depth_multiple),shortcut=False)
        logger.info("Layer 15 execute {}".format(c15.shape))
        
        c16 = Conv(c15, (3,3,int(256*self.width_multiple),int(256*self.width_multiple)),True,downsample=True)
        logger.info("Layer 16 execute {}".format(c16.shape))

        c17 = tf.concat([c16,c12],3)
        logger.info("Layer 17 execute {}".format(c17.shape))

        shape = c17.shape
        c18 = C2F(c17,shape[-1].value, int(512*self.width_multiple) ,int(3*self.depth_multiple),shortcut=False)
        logger.info("Layer 18 execute {}".format(c18.shape))


        c19 = Conv(c18, (3,3,int(512*self.width_multiple),int(512*self.width_multiple)),True,downsample=True)
        logger.info("Layer 19 execute {}".format(c19.shape))

        c20 = tf.concat([c19,c9],3)
        logger.info("Layer 20 execute {}".format(c20.shape))

        shape = c20.shape
        c21 = C2F(c20,shape[-1].value, int(512*self.width_multiple*self.ratio) ,int(3*self.depth_multiple),shortcut=False)
        logger.info("Layer 21 execute {}".format(c21.shape))

        return c15,c18,c21
    
    def Detect():
        return


    def Network(self):
        try:
            c4,c6,c9 = self.Backbone()
            c14,c18,c21 = self.Head(c4,c6,c9)
        except:
            logger.error(traceback.print_exc)
            return None
        return c14,c18,c21
    




if __name__=="__main__":

    batch_size = 2
    height = 640
    width = 640
    channels = 3
    x = tf.random.uniform([height*width*channels*batch_size],0,1)
    input_data = tf.reshape(x,[batch_size,width,height,channels])

    yolov8 = YOLOV8(input_data=input_data)

    routes = yolov8.Network()
    if routes==None:
         logger.error("Error in Model Buidling")
