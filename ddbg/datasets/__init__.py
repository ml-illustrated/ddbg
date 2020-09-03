from .mnist import MNISTWithIDs
from .fashion_mnist import FashionMNISTWithIDs
from .cifar10 import CIFAR10WithIDs

dataset_name__dataset_class = dict(
    mnist = MNISTWithIDs,
    fashion_mnist = FashionMNISTWithIDs,
    cifar10 = CIFAR10WithIDs,
)
