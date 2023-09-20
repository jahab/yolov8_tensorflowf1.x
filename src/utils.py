import tensorflow as tf
tf.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
import os
import os.path as osp
import cv2
class Train:
    def __init__(self) -> None:
        pass


class Loss:
    def __init__(self) -> None:
        pass

class Dataloader:

    """
    DatasetPath
    |____labels
    |    |____train
    |    |____val
    |
    |____images
    |    |____train
    |____|____val
    
    """
    def __init__(self, dataset_path, train=True,validation=False, test=False) -> None:
        self.dataset_path =  dataset_path
        if train:
            assert(osp.exists(osp.join(dataset_path,"labels/train"))==True,"TRaining Labels path doesnot exist")
            self.image_path = osp.join(dataset_path,"images/train")
            self.labels_path = osp.join(dataset_path,"labels/train")
            self.images_list = list(map(lambda x: osp.join(self.image_path,x),os.listdir(self.image_path)))
            self.label_list = list(map(lambda x: osp.join(self.labels_path,x),os.listdir(self.labels_path)))


        if validation:
            assert(osp.exists(osp.join(dataset_path,"labels/val"))==True,"Validation Labels path doesnot exist")
            self.image_path = osp.join(dataset_path,"images/val")
            self.labels_path = osp.join(dataset_path,"labels/val")
            

        # Create a dataset from file paths
        
        self.dataset = tf.data.Dataset.from_tensor_slices(self.images_list)
        self.dataset = self.dataset.map(self.parse_func)
        self.dataset = self.dataset.batch(2)
        # Create an iterator for the dataset
        self.iterator = self.dataset.make_one_shot_iterator()
    
    def __next__(self):
        return self.iterator.get_next()

    def parse_func(self,filename):
        out = tf.py_function(self.load_and_preprocess_image, [filename], tf.float32)
        return out
    
    def load_and_preprocess_image(self,file_path):
        # Load the image file
        path = file_path.numpy()[0].decode("utf-8") # .numpy() retrieves data from eager tensor
        img = cv2.imread(path)
        img = cv2.resize(img,(224,224))
        img = tf.math.divide(img,255)
        return img

if __name__=="__main__":
    # dataloader = Dataloader("/home/jafar/Desktop/ultralytics/ultralytics/yolo/data/datasets/coco128", train=True)
    # batch_size = 2
    # next_batch = dataloader.iterator.get_next()
    
    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             batch_images = sess.run(next_batch)
    #             # Process the batch of images (e.g., pass through a model)
    #             print("Batch shape:", batch_images.shape)
    #         except tf.errors.OutOfRangeError:
    #             break

    file_paths = ["/home/jafar/Desktop/ultralytics/ultralytics/yolo/data/datasets/coco128/images/train2017/000000000009.jpg", "/home/jafar/Desktop/ultralytics/ultralytics/yolo/data/datasets/coco128/images/train2017/000000000030.jpg", "/home/jafar/Desktop/ultralytics/ultralytics/yolo/data/datasets/coco128/images/train2017/000000000034.jpg"]

    # Create a dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)



def load_and_preprocess_image(path):
    # print(path.value)
    path = path.numpy()[0].decode("utf-8") # .numpy() retrieves data from eager tensor
    img = cv2.imread(path)
    img = cv2.resize(img,(224,224))
    img = tf.math.divide(img,255)
    return img


def parse_func(filename):
    out = tf.py_function(load_and_preprocess_image, [filename], tf.float32)
    return out

file_paths = ["/home/jafar/Desktop/ultralytics/ultralytics/yolo/data/datasets/coco128/images/train2017/000000000009.jpg", "/home/jafar/Desktop/ultralytics/ultralytics/yolo/data/datasets/coco128/images/train2017/000000000030.jpg", "/home/jafar/Desktop/ultralytics/ultralytics/yolo/data/datasets/coco128/images/train2017/000000000034.jpg"]

# Create a dataset from file paths
dataset = tf.data.Dataset.from_tensor_slices(file_paths)
dataset = dataset.batch(2)
dataset = dataset.map(parse_func)
iterator = dataset.make_one_shot_iterator()
sess = tf.Session()
print(sess.run(iterator.get_next()))

dataset = tf.data.Dataset.list_files("/home/jafar/Desktop/ultralytics/ultralytics/yolo/data/datasets/coco128/images/train2017/*.jpg")
    # Define a function to load and preprocess image files
def load_and_preprocess_image(file_path):
    # Load the image file
    # print("!!!!!!!!!!!!", file_path)
    print("!!!!!!!!!!!!", file_path.numpy()[0].decode('utf-8'))
    # print("file_path: ",bytes.decode(file_path.numpy()))
    # image = tf.io.read_file(file_path)
    # print(image.shape)
    # Decode the image (e.g., JPEG or PNG)
    # image = tf.image.decode_image(image, channels=3)
    # Resize the image to a consistent shape (e.g., 224x224)
    # image = tf.image.resize(image, [224, 224])
    # Normalize pixel values to the range [0, 1]
    image = cv2.imread(file_path)
    image = cv2.resize(image,(224,224))
    # image = tf.cast(image, tf.float32) / 255.0
    # print(image)
    return image

    # load_and_preprocess_image(file_paths[1])
    # Apply the load_and_preprocess_image function to each file path in the dataset
    # dataset = dataset.map(load_and_preprocess_image)

dataset = dataset.map(load_and_preprocess_image)
    # dataset = dataset.map(lambda x: x + x)
    # Batch the dataset with a specified batch size
batch_size = 2
dataset = dataset.batch(batch_size)
print(list(dataset))
# Create an iterator for the dataset
iterator = dataset.make_one_shot_iterator()

# Get the next batch of images
next_batch = iterator.get_next()

    # Start a session and fetch the next batch of images
    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             batch_images = sess.run(next_batch)
    #             # Process the batch of images (e.g., pass through a model)
    #             print("Batch shape:", batch_images.shape)
    #         except tf.errors.OutOfRangeError:
    #             break