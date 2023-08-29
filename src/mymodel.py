import tensorflow as tf
tf.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

import numpy as np 
import logging
import traceback
from modules import *
# Configure the logger
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Create a logger instance
logger = logging.getLogger(__name__)


# def Conv(input_data, filters_shape:tuple, trainable:bool, name=None, downsample:bool=False, padding:str="SAME",  activate:bool=True, bn:bool=True, act_fun='leaky_relu'):
    
#     """
#     params: 
#     input_data: input_tensor
#     filters_shape: 1D tuple of length 4, [kernerl_h,kernel_w,channel_in, channel_out]
#     trainable: true/false
#     """
#     # with tf.variable_scope(name):
#     if downsample:
#         pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
#         paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
#         input_data = tf.pad(input_data, paddings, 'CONSTANT')
#         strides = (2, 2)
#         padding="VALID"
#     else:
#         strides = (1, 1)
        


#     conv_layer = tf.layers.Conv2D(filters=filters_shape[-1],
#                             kernel_size=(filters_shape[0],filters_shape[1]), 
#                             strides=strides, 
#                             padding=padding,
#                             trainable=trainable,
#                             use_bias=True,
#                             kernel_initializer = tf.random_normal_initializer(stddev=0.01),
#                             name=name)

#     if bn:
#         bn_layer = tf.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
#                                         beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(),
#                                         moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer(),
#                                         beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
#                                         trainable=trainable, name=None, renorm=False, renorm_clipping=None,
#                                         renorm_momentum=0.99)

#     res = conv_layer(input_data)
#     res = bn_layer(res)

#     if activate:
#         res = tf.nn.leaky_relu(res)

#     return res

# def Bottleneck(input_data,ch_in,ch_out, k=(3,3), shortcut=False):
    
#     kernel_shape=(*k,ch_in,ch_out)
#     conv1 = Conv(input_data,kernel_shape,True)
    
#     # logger.info("in bottle nek----{}".format(conv1.shape))
#     conv2 = Conv(conv1,kernel_shape,True)
#     # logger.info("in bottle nek----{}".format(conv2.shape))
    
#     if shortcut:
#         return conv2
#     else:
#         # logger.info("concat shape {}".format(tf.math.add(conv2,input_data).shape))
#         return tf.math.add(conv2,input_data)


# def C2F(input_data,ch_in:int,ch_out:int, replicate:int, shortcut:bool=False,e=0.5):

#     c = int(ch_out * e)  # hidden channels
#     # logger.info("First conv in C2F {} {} {}".format(ch_in, 2 * c, e))
#     kernel_shape1 = (1,1,ch_in, 2 * c)
#     # logger.info("__{} {}".format(input_data.shape, kernel_shape1))
#     conv1 = Conv(input_data, kernel_shape1,True)
    
#     x = conv1
#     splits = tf.split(x,2,3)
#     # logger.info("------------",splits, len(splits))
#     for i in range(replicate):
#         temp = Bottleneck(splits[-1],c,c,shortcut=shortcut)
#         # logger.info("temp shape {}".format(temp.shape))
#         splits.extend([temp])


#     concat = tf.concat(splits,3)

#     kernel_shape2 = (1,1,(2 + replicate) * c, ch_out)

#     conv2 =  Conv(concat,kernel_shape2,True)
#     return conv2
    

# def SPPF(input_data,ch_in,ch_out, k=(1,1)):
#     kernel_shape=(*k,ch_in,ch_out)
#     c1 = Conv(input_data,kernel_shape,True,downsample=False)
#     maxpool1 = tf.layers.MaxPooling2D(5,1, padding="same")
#     m1 = maxpool1(c1)
#     m2 = maxpool1(m1)
#     m3 = maxpool1(m2)
#     concat = tf.concat([c1,m1,m2,m3],3)
#     kernel_shape2 = (1,1,ch_out)
#     c2 = Conv(concat,kernel_shape2,True)
#     return c2


class MyModel(tf.keras.Model):
    def __init__(self, d:float=0.33,w:float=0.50,r:float=2.0, batch_size:int=4,input_shape:tuple=(640,640,3)):
        super(MyModel,self).__init__()
        self.depth_multiple = d
        self.width_multiple = w
        self.ratio = r
        self.batch_size = batch_size
        self.input_data = None
        
    def Backbone(self):

        k1 = (3,3,3,int(64*self.width_multiple))
        self.c0 = Conv(k1,True,downsample=True)
        
        k1 = (3,3,int(64*self.width_multiple),int(128*self.width_multiple))
        self.c1 = Conv(k1,True,downsample=True)
                
        self.c2 = C2F(int(128*self.width_multiple),int(128*self.width_multiple),int(3*self.depth_multiple),True)
        
        k1 = (3,3,int(128*self.width_multiple),int(256*self.width_multiple))
        self.c3 = Conv(k1,True,downsample=True)
        
        self.c4 = C2F(int(256*self.width_multiple),int(256*self.width_multiple),int(6*self.depth_multiple),True)
        
        k1 = (3,3,int(256*self.width_multiple),int(512*self.width_multiple))
        self.c5 = Conv(k1,True,downsample=True)
        
        self.c6 = C2F(int(512*self.width_multiple),int(512*self.width_multiple),int(6*self.depth_multiple),True)
        
        k1 = (3,3,int(512*self.width_multiple),int(512*self.width_multiple*self.ratio))
        self.c7 = Conv(k1,True,downsample=True)
        
        self.c8 = C2F(int(512*self.width_multiple*self.ratio),int(512*self.width_multiple*self.ratio),int(3*self.depth_multiple),True)
       
        self.c9 = SPPF(int(512*self.width_multiple*self.ratio),int(512*self.width_multiple*self.ratio))

    def Head(self):
        self.c10 = Upsample(size=(40,40))
        self.c11 = Concat(axis=3)
        
        shape = self.c9.output_filter + self.c6.output_filter
        self.c12 = C2F(shape[-1].value, int(512*self.width_multiple),int(3*self.depth_multiple),shortcut=False)
        
        self.c13 = Upsample(size=(80,80))

        self.c14 = Concat(axis=3)
        shape = self.c12.output_filter+self.c4.output_filter
        self.c15 = C2F(shape[-1].value, int(256*self.width_multiple) ,int(3*self.depth_multiple),shortcut=False)
        
        self.c16 = Conv((3,3,int(256*self.width_multiple),int(256*self.width_multiple)),True,downsample=True)
        
        self.c17 = Concat(axis=3)

        shape = self.c12.output_filter+self.c16.output_filter
        self.c18 = C2F(shape[-1].value, int(512*self.width_multiple) ,int(3*self.depth_multiple),shortcut=False)
        
        self.c19 = Conv((3,3,int(512*self.width_multiple),int(512*self.width_multiple)),True,downsample=True)
        
        self.c20 = Concat(axis=3)

        shape = self.c9.output_filter + self.c19.output_filter
        self.c21 = C2F(shape[-1].value, int(512*self.width_multiple*self.ratio) ,int(3*self.depth_multiple),shortcut=False)

    def Detect(self,ch:tuple=(),nc=80):
        print("-------------",ch)
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = tf.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels

        self.conv_layer1 = tf.layers.Conv2D(filters=4*self.reg_max,
                            kernel_size=(1,1), 
                            strides=1, 
                            padding="VALID",
                            trainable=True,
                            use_bias=True,
                            kernel_initializer = tf.random_normal_initializer(stddev=0.01))

        self.conv_layer2 = tf.layers.Conv2D(filters=self.nc,
                            kernel_size=(1,1), 
                            strides=1, 
                            padding="VALID",
                            trainable=True,
                            use_bias=True,
                            kernel_initializer = tf.random_normal_initializer(stddev=0.01))
        
        bbox_loss = {}
        cls_loss = {}

        for i,dim in enumerate(ch):
            
            y1 = Conv((3,3,dim,c2),trainable=True,padding="SAME",downsample=False)
            y2 = Conv((3,3,y1.output_filter,c3),trainable=True,padding="SAME",downsample=False)
            conv_layer1 = tf.layers.Conv2D(filters=4*self.reg_max,
                            kernel_size=(1,1), 
                            strides=1, 
                            padding="VALID",
                            trainable=True,
                            use_bias=True,
                            kernel_initializer = tf.random_normal_initializer(stddev=0.01))
            bbox_loss[i] = [y1,y2,conv_layer1]

            y1 = Conv((3,3,dim,c2),trainable=True,padding="SAME",downsample=False)
            y2 = Conv((3,3,y1.output_filter,c3),trainable=True,padding="SAME",downsample=False)
            conv_layer2 = tf.layers.Conv2D(filters=self.nc,
                            kernel_size=(1,1), 
                            strides=1, 
                            padding="VALID",
                            trainable=True,
                            use_bias=True,
                            kernel_initializer = tf.random_normal_initializer(stddev=0.01))
            cls_loss[i] = [y1,y2,conv_layer2]


    def __call__Backbone(self,input_data):
        x0 = self.c0(input_data)
        logger.info("layer 0 execute {}".format(x0.shape))
        x1 = self.c1(x0)
        logger.info("layer 1 execute {}".format(x1.shape))
        x2 = self.c2(x1)
        logger.info("layer 2 execute {}".format(x2.shape))
        x3 = self.c3(x2)
        logger.info("layer 3 execute {}".format(x3.shape))
        x4 = self.c4(x3)
        logger.info("Layer 4 execute {}".format(x4.shape))
        x5 = self.c5(x4)
        logger.info("layer 5 execute {}".format(x5.shape))
        x6 = self.c6(x5)
        logger.info("Layer 6 execute {}".format(x6.shape))
        x7 = self.c7(x6)
        logger.info("layer 7 execute {}".format(x7.shape))
        x8 = self.c8(x7)
        logger.info("Layer 8 execute {}".format(x8.shape))
        x9 = self.c9(x8)
        logger.info("Layer 9 execute {}".format(x9.shape))

        return x4, x6, x9

    def __call__Head(self,x4,x6,x9):
        x10 = self.c10(x9)
        logger.info("Layer 10 execute {}".format(x10.shape))
        x11 = self.c11([x10,x6])
        logger.info("Layer 11 execute {}".format(x11.shape))
        x12 = self.c12(x11)
        logger.info("Layer 12 execute {}".format(x12.shape))
        x13 = self.c13(x12)
        logger.info("Layer 13 execute {}".format(x13.shape))
        x14 = self.c14([x13,x4])
        logger.info("Layer 14 execute {}".format(x14.shape))
        x15 = self.c15(x14)
        logger.info("Layer 15 execute {}".format(x15.shape))
        x16 = self.c16(x15)
        logger.info("Layer 16 execute {}".format(x16.shape))
        x17 = self.c17([x16,x12])
        logger.info("Layer 17 execute {}".format(x17.shape))
        x18 = self.c18(x17)
        logger.info("Layer 18 execute {}".format(x18.shape))
        x19 = self.c19(x18)
        logger.info("Layer 19 execute {}".format(x19.shape))
        x20 = self.c20([x19,x9])
        logger.info("Layer 20 execute {}".format(x20.shape))
        x21 =self.c21(x20)
        logger.info("Layer 21 execute {}".format(x21.shape))
        return x15,x18,x21

class YOLOV8(tf.keras.Model):
    def __init__(self, d:float=0.33,w:float=0.50,r:float=2.0, batch_size:int=4,input_shape:tuple=(640,640,3)) -> None:
        super(YOLOV8, self).__init__()
        self.depth_multiple = d
        self.width_multiple = w
        self.ratio = r
        self.batch_size = batch_size
        # self.input_shape = input_shape
        self.input_data = None



    def Backbone(self,input_data):
        try:
            k1 = (3,3,3,int(64*self.width_multiple))
            c0 = Conv(input_data,k1,True,downsample=True)
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
    
    def Detect(self,x, ch:tuple=(),nc=80):
        print("-------------",ch)
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = tf.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels

        conv_layer1 = tf.layers.Conv2D(filters=4*self.reg_max,
                            kernel_size=(1,1), 
                            strides=1, 
                            padding="VALID",
                            trainable=True,
                            use_bias=True,
                            kernel_initializer = tf.random_normal_initializer(stddev=0.01))

        conv_layer2 = tf.layers.Conv2D(filters=self.nc,
                            kernel_size=(1,1), 
                            strides=1, 
                            padding="VALID",
                            trainable=True,
                            use_bias=True,
                            kernel_initializer = tf.random_normal_initializer(stddev=0.01))


        for i,dim in enumerate(ch):
            y1 = Conv(x[i],(3,3,dim,c2),trainable=True,padding="SAME",downsample=False)
            y1 = Conv(y1,(3,3,y1.shape[-1].value,c3),trainable=True,padding="SAME",downsample=False)
            y1 = conv_layer1(y1)

            y2 = Conv(x[i],(3,3,dim,c2),trainable=True,padding="SAME",downsample=False)
            y2 = Conv(y2,(3,3,y2.shape[-1].value,c3),trainable=True,padding="SAME",downsample=False)
            y2 = conv_layer2(y2)

            x[i] = tf.concat([y1,y2],3)
        return x


    def __call__(self,input_data):
        try:
            c4,c6,c9 = self.Backbone(input_data)
            c15,c18,c21 = self.Head(c4,c6,c9)
            ch = c15.shape[-1].value,c18.shape[-1].value,c21.shape[-1].value
            detect = self.Detect([c15,c18,c21], ch = ch, nc=80)
            print("DETECT SHAPE-------",detect[0].shape,detect[1].shape,detect[2].shape)
            flatten = tf.layers.Flatten()
            fc2 = tf.layers.Dense(4, activation=None)
            
            return fc2(flatten(detect[0]))

        except:
            logger.error(traceback.print_exc())
            return None
    


if __name__=="__main__":

    num_examples = 100
    height = 640
    width = 640
    channels = 3
    batch_size = 4
    # x = tf.random.uniform([height*width*channels*num_examples],0,1)
    input_data = tf.random.normal((num_examples,) + (height,width,channels))
    # samples = tf.random.categorical(tf.math.log([[0.5, 0.5]]), num_examples)
    samples = tf.random.uniform((num_examples,4), minval=0, maxval=3, dtype=tf.int32)
    # input_data = tf.reshape(x,[num_examples,width,height,channels])

    mymodel = MyModel()
    mymodel(input_data)



    # yolov8 = YOLOV8()
    
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    # dataset = tf.data.Dataset.from_tensor_slices((input_data, samples))
    # dataset = dataset.shuffle(buffer_size=num_examples).batch(batch_size).prefetch(batch_size)
    
    # num_epochs = 10
    # tf.global_variables_initializer()
    # for epoch in range(num_epochs):
    #     for batch_x, batch_y in dataset:
            
    #         with tf.GradientTape() as tape:
    #             logits = yolov8(batch_x)
    #             loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))
    #             print("ssssssssssssss",loss,yolov8.variables)
    #         gradients = tape.gradient(loss, yolov8.variables)
    #         optimizer.apply_gradients(zip(gradients, yolov8.variables))

    #     # Calculate accuracy on validation set
    #     # correct_prediction = tf.equal(tf.argmax(yolov8(mnist.validation.images), 1), tf.argmax(mnist.validation.labels, 1))
    #     # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     # val_accuracy = accuracy.numpy()
    #     print("Epoch {}".format(epoch + 1))
