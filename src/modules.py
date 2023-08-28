import tensorflow as tf
import numpy as np


tf.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

class Conv(tf.keras.Model):
    def __init__(self,filters_shape:tuple, 
                        trainable:bool, 
                        name:str=None, 
                        downsample:bool=False, 
                        padding:str="SAME",  
                        activate:bool=True, 
                        bn:bool=True,
                        act_fun:str='leaky_relu') -> None:
        
        super().__init__()
        self.activate = activate
        self.act_fun = act_fun
        self.downsample = downsample
        self.filters_shape = filters_shape
        if self.downsample:
            strides = (2, 2)
            padding="VALID"
        else:
            strides = (1, 1)
        


        self.conv_layer = tf.layers.Conv2D(filters=self.filters_shape[-1],
                                kernel_size=(self.filters_shape[0],self.filters_shape[1]), 
                                strides=strides, 
                                padding=padding,
                                trainable=trainable,
                                use_bias=True,
                                kernel_initializer = tf.random_normal_initializer(stddev=0.01),
                                name=name)

        if bn:
            self.bn_layer = tf.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                            beta_initializer=tf.zeros_initializer(), gamma_initializer=tf.ones_initializer(),
                                            moving_mean_initializer=tf.zeros_initializer(), moving_variance_initializer=tf.ones_initializer(),
                                            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                                            trainable=trainable, name=None, renorm=False, renorm_clipping=None,
                                            renorm_momentum=0.99)
        
    def __call__(self,input_data):
        if self.downsample:
            pad_h, pad_w = (self.filters_shape[0] - 2) // 2 + 1, (self.filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
        x = self.conv_layer(input_data)
        x = self.bn_layer(x)

        if self.activate:
            if self.act_fun =="leaky_relu":
                x = tf.nn.leaky_relu(x)

        return x


class Bottleneck(tf.keras.Model):
    def __init__(self, ch_in:int,
                    ch_out:int, 
                    k:tuple=(3,3), 
                    shortcut:bool=False):
        super(Bottleneck,self).__init__()
        self.kernel_shape=(*k,ch_in,ch_out)
        self.conv1 = Conv(self.kernel_shape,True)
        self.conv2 = Conv(self.kernel_shape,True)
        self.shortcut = shortcut

    def __call__(self,input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        
        if self.shortcut:            
            return x
        else:
            return tf.math.add(x,input_data)

class C2F(tf.keras.Model):
    def __init__(self, ch_in:int,
                 ch_out:int, 
                 replicate:int, 
                 shortcut:bool=False,
                 e:float=0.5):
        super(C2F,self).__init__()
        
        self.c = int(ch_out * e)  # hidden channels
        kernel_shape1 = (1,1,ch_in, 2 * self.c)
        self.conv1 = Conv(kernel_shape1,True)
        self.shortcut = shortcut
        self.bottlneck_list = []
        for i in range(replicate):
            self.bottlneck_list.append(Bottleneck(self.c,self.c,shortcut=self.shortcut))
        
        kernel_shape2 = (1,1,(2 + replicate) * self.c, ch_out)
        self.conv2 = Conv(kernel_shape2,True)
    
    def __call__(self,input_data):
        x = self.conv1(input_data)
        splits = tf.split(x,2,3)
        
        for m in self.bottlneck_list:
            temp = m(splits[-1])
            splits.extend([temp])
        
        concat = tf.concat(splits,3)
        return self.conv2(concat)

class SPPF(tf.keras.Model):
    def __init__(self, ch_in:int,
                    ch_out:int,
                    k:tuple=(1,1)):
        
        super(SPPF,self).__init__()
        self.kernel_shape = (*k,ch_in,ch_out)
        self.conv1 = Conv(self.kernel_shape,True,downsample=False)
        self.maxpool1 = tf.layers.MaxPooling2D(5,1, padding="same")
        self.kernel_shape2 = (1,1,ch_out)
        self.conv2 = Conv(self.kernel_shape2,True)
    
    def __call__(self,input_data):
        x = self.conv1(input_data)
        x1 = self.maxpool1(x)
        x2 = self.maxpool1(x)
        x3 = self.maxpool1(x)
        concat = tf.concat([x,x1,x2,x3],3)
        return self.conv2(concat)

class Upsample():
    def __init__(self,size:tuple=(40,40)):
        self.size = size
        pass
    def __call__(self,input_data):
        x = tf.image.resize_images(input_data, *self.size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return x