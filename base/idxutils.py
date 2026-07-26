from typing import override

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

np.seterr(all="raise")

__all__ = ["Idx1", "Idx3"]


class Idx1:
    """
    A minimal class to handle IO operations using Idx1 files

    IDX1 file format:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    """

    def __init__(self, filepath: str) -> None:
        """
        `Parameters`:
        filepath: str - path to unzipped Idx1 resource

        `Returns`:
        None

        `Notes`:
        Depends on NumPy
        """

        try:
            with open(file=filepath, mode="rb") as fp:
                ubytes: NDArray[np.uint8] = np.fromfile(fp, dtype=np.uint8)
        except FileNotFoundError as fnf_error:
            raise RuntimeError(f"Creation of Idx1 object from file {filepath} failed!") from fnf_error

        self.magic: int = int.from_bytes(ubytes[:4], byteorder="big")  # Idx1 magic number
        self.count: int = int.from_bytes(ubytes[4:8], byteorder="big")  # count of the data elements (labels)

        assert self.count == ubytes.size - 8, "There seems to be a parsing error or the binary file is corrupted!"
        # the actual data
        # type casting the data from np.uint8 to np.float64 since np.exp() raises FloatingPointError with np.uint8 arrays
        self.data: NDArray[np.float64] = ubytes[8:].astype(np.float64)

    @override
    def __repr__(self) -> str:
        return f"Idx1 object(magic: {self.magic}, count: {self.count:,})"

    def __getitem__(self, index: int) -> np.float64:
        """
        `Parameters`:
        index: int - offset of the element to return from the labels array.

        `Returns`:
        np.float64 - the index th element in the labels array

        `Notes`:
        IndexErrors are left for NumPy to handle.
        """

        return self.data[index]

    def __setitem__(self, index: int) -> None:
        raise PermissionError("Idx1 objects are immutable!")


class Idx3:
    """
    A minimal class to handle IO operations using Idx3 files

    IDX3 file format:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    """

    def __init__(self, filepath: str) -> None:
        """
        `Parameters`:
        filepath: str - path to unzipped Idx3 resource

        `Returns`:
        None

        `Notes`:
        Depends on NumPy
        """

        try:
            with open(file=filepath, mode="rb") as fp:
                ubytes: NDArray[np.uint8] = np.fromfile(fp, dtype=np.uint8)
        except FileNotFoundError as fnf_error:
            raise RuntimeError(f"Creation of Idx3 object from file {filepath} failed!") from fnf_error

        self.magic: int = int.from_bytes(ubytes[:4], byteorder="big")  # Idx3 magic number
        self.count: int = int.from_bytes(ubytes[4:8], byteorder="big")  # count of the data elements (images)

        # shape is shape of each element NOT the shape of the overall data
        self.shape: tuple[int, int] = (int.from_bytes(ubytes[8:12], byteorder="big"), int.from_bytes(ubytes[12:16], byteorder="big"))

        # supposed to be 28 x 28 for MNIST inspired datasets.
        pixels_per_image: int = self.shape[0] * self.shape[1]  # pixels per image (28 x 28)
        assert (self.count * pixels_per_image) == (ubytes.size - 16), "There seems to be a parsing error or the binary file is corrupted!"

        # the actual data
        # Idx3 file stores data as bytes but we'll load in each byte as a 64 bit double
        # because np.exp() raises a FloatingPointError with np.uint8 type arrays
        self.data: NDArray[np.float64] = ubytes[16:].reshape(self.count, pixels_per_image).T.astype(np.float64)

    @override
    def __repr__(self) -> str:
        return f"Idx3 object (magic: {self.magic}, shape: {self.shape}, count: {self.count:,})"

    def __getitem__(self, index: int) -> NDArray[np.float64]:
        """
        `Parameters`:
        index: int - the column to return from the matrix.

        `Returns`:
        NDArray[np.float64] - all pixels of index th image, i.e returns the index th column of the transposed matrix

        `Notes`:
        IndexErrors are left for NumPy to handle.
        """

        return self.data[:, index]

    def __setitem__(self, index: int) -> None:
        raise PermissionError("Idx3 objects are immutable!")


def peek_idx(idx3: str, idx1: str, image_dim: tuple[int, int] = (28, 28), colormap: str = "binary") -> None:
    """
    Samples 15 randomly choosen pairs of images and labels from the provided Idx resources and plots them.

    `Parameters`:
    Idx3: str - the path to the Idx3 resource
    Idx1: str - path to corresponding Idx1 resource
    Idx3elem_dim: typing.Tuple[int, int] - shape of the image matrices in the Idx3 file
    colormap: str - matplotlib cmap specification to use, when plotting the images

    `Returns`:
    None

    `Notes`:
    Depends on NumPy and matplotlib.
    """

    images = Idx3(idx3)
    labels = Idx1(idx1)
    assert images.count == labels.count, (
        f"Mismatch in the number of elements stored in Idx1 and Idx3 files:: Idx1: {labels.count}, Idx3: {images.count}"
    )

    nrows, ncols = image_dim
    fig, axes = plt.subplots(nrows=1, ncols=15)
    fig.set_size_inches(20, 4)
    random_indices: NDArray[np.int64] = np.random.choice(np.arange(0, images.count), size=15, replace=False)
    for ax, idx in zip(axes, random_indices):
        ax.imshow(images[idx].reshape(nrows, ncols), cmap=colormap)
        ax.set_title(f"{labels[idx]:1.0f}")  # remember the labels are now np.float64 s
        ax.set_xticks([])  # remove the x axis ticks and labels
        ax.set_yticks([])  # remove the y axis ticks and labels
    plt.show()  # type: ignore
