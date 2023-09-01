from typing import Any
import math
import tensorflow as tf
import numpy as np


tf.enable_eager_execution(config=None, device_policy=None, execution_mode=None)


def ModuleList2Seq():
    def __init__(self,layer_list:list) -> None:
        self.layer_list = layer_list
        

    def __call__(self, inputs) -> Any:
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


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
        self.input_filter = self.filters_shape[-2]
        self.output_filter = self.filters_shape[-1]
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
        self.input_filter = self.conv1.input_filter
        self.output_filter = self.conv2.output_filter

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
        self.concat1 = Concat(axis=3)
        self.input_filter = self.conv1.input_filter
        self.output_filter = self.conv2.output_filter
    
    def __call__(self,input_data):
        x = self.conv1(input_data)
        splits = tf.split(x,2,3)
        
        for m in self.bottlneck_list:
            temp = m(splits[-1])
            splits.extend([temp])
        
        concat = self.concat1(splits) #tf.concat(splits,3)
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
        self.concat1 = Concat(axis=3)
        self.input_filter = self.conv1.input_filter
        self.output_filter = self.conv2.output_filter
    def __call__(self,input_data):
        x = self.conv1(input_data)
        x1 = self.maxpool1(x)
        x2 = self.maxpool1(x)
        x3 = self.maxpool1(x)
        concat = self.concat1([x,x1,x2,x3])
        return self.conv2(concat)

class Upsample():
    def __init__(self,size:tuple=(40,40)):
        self.size = size
        pass
    def __call__(self,input_data):
        x = tf.image.resize_images(input_data, *self.size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return x

class Concat():
    def __init__(self,axis:int=3) -> None:
        self.axis = axis
        
    def __call__(self,x:list):
        return tf.concat(x,self.axis)

class DFL(tf.keras.Model):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        
        # self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv = tf.layers.Conv2D(1,
                            kernel_size=(1,1), 
                            strides=1, 
                            padding="VALID",
                            trainable=True,
                            use_bias=False,
                            kernel_initializer = tf.random_normal_initializer(stddev=0.01))
        

        self.conv.set_weights = np.arange(c1,dtype=float)
        # x = torch.arange(c1, dtype=torch.float)
        # self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        
        self.c1 = c1

    def __call__(self, x):
        b, c, a = x.shape  # batch, channels, anchors 
        x = tf.reshape(x,[b, 4, self.c1, a]) # TODO: find out the shape and adjust
        x = tf.image.transpose(x) # TODO: find out the shape and adjust
        x = tf.nn.softmax(x,axis=1) # TODO: find out the shape and adjust
        x = self.conv(x) # TODO: find out the shape and adjust
        x = tf.reshape(x, [b,4,a]) # TODO: find out the shape and adjust
        return x
        # return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Detect(tf.keras.Model):
    def __init__(self, ch:tuple=(),nc=80):
        super().__init__()
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
        
        self.concat = Concat(axis=3)
        self.bbox_layer = {}
        self.cls_layer = {}

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
            self.bbox_layer[i] = ModuleList2Seq([y1,y2,conv_layer1])

            y1 = Conv((3,3,dim,c2),trainable=True,padding="SAME",downsample=False)
            y2 = Conv((3,3,y1.output_filter,c3),trainable=True,padding="SAME",downsample=False)
            conv_layer2 = tf.layers.Conv2D(filters=self.nc,
                            kernel_size=(1,1), 
                            strides=1, 
                            padding="VALID",
                            trainable=True,
                            use_bias=True,
                            kernel_initializer = tf.random_normal_initializer(stddev=0.01))
            self.cls_layer[i] = ModuleList2Seq([y1,y2,conv_layer2])

    def __call__(self,x):
        for i in range(self.nl):
            y1 = self.bbox_layer[i](x[i])
            y2 = self.cls_layer[i](x[i])
            x[i] = self.concat([y1,y2])
        return x
    
    def bias_init(self):
        # TODO
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(self.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)