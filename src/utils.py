import tensorflow as tf

class Train:
    def __init__(self) -> None:
        pass


class Loss:
    def __init__(self) -> None:
        pass

class Dataloader:
    def __init__(self, dataset_path, train=True,validation=False, test=False) -> None:
        self.dataset_path =  dataset_path
        assert()

        # Create a dataset from file paths
        dataset = tf.data.Dataset.from_tensor_slices(dataset_path)