import sys

from base.idxutils import Idx1, Idx3
from base.nnet import NNetworkMinimal


def main() -> None:
    mnist_train_images = Idx3(r"../MNIST/train-images.idx3-ubyte")
    mnist_test_images = Idx3(r"../MNIST/t10k-images.idx3-ubyte")
    mnist_train_labels = Idx1(r"../MNIST/train-labels.idx1-ubyte")
    mnist_test_labels = Idx1(r"../MNIST/t10k-labels.idx1-ubyte")

    model_0 = NNetworkMinimal(nodes_in=784, nodes_hid=10, nodes_out=10, alpha=0.125, maxiterations=10_000)
    model_0.fit(mnist_train_images.data, mnist_train_labels.data)

    print(f"[[MNIST]] :: train set accuracy score = {model_0.accuracy_score(mnist_train_images.data, mnist_train_labels.data):.5f}")
    print(f"[[MNIST]] :: test set accuracy score = {model_0.accuracy_score(mnist_test_images.data, mnist_test_labels.data):.5f}")

    model_0.save(r"./models/mnist-10000")

    fashion_mnist_train_images = Idx3(r"../Fashion-MNIST/train-images-idx3-ubyte")
    fashion_mnist_test_images = Idx3(r"../Fashion-MNIST/t10k-images-idx3-ubyte")
    fashion_mnist_train_labels = Idx1(r"../Fashion-MNIST/train-labels-idx1-ubyte")
    fashion_mnist_test_labels = Idx1(r"../Fashion-MNIST/t10k-labels-idx1-ubyte")

    model_1 = NNetworkMinimal(nodes_in=784, nodes_hid=10, nodes_out=10, alpha=0.1, maxiterations=10_000)
    model_1.fit(fashion_mnist_train_images.data, fashion_mnist_train_labels.data)

    print(
        f"[[Fashion-MNIST]] :: train set accuracy score = {model_1.accuracy_score(fashion_mnist_train_images.data, fashion_mnist_train_labels.data):.5f}"
    )
    print(
        f"[[Fashion-MNIST]] :: test set accuracy score = {model_1.accuracy_score(fashion_mnist_test_images.data, fashion_mnist_test_labels.data):.5f}"
    )

    model_1.save(r"./models/fashion-mnist-10000")

    sys.exit(0)


if __name__ == "__main__":
    main()
