from typing import override

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

__all__: tuple[str, ...] = ("IdxDataset",)


class IdxDataset(Dataset[torch.Tensor]):
    """
    A PyTorch compatible class to handle IO and iteration operations with idx1 (labels) and Idx3 (images) files

    IDX1 file format:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label

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

    def __init__(self, idx1_filepath: str, idx3_filepath: str) -> None:
        """
        `Parameters`:
        idx1_filepath: str - path to unzipped idx1 resource (labels)
        idx3_filepath: str - path to unzipped idx3 resource (images)

        `Returns`:
        None

        `Notes`:
        Depends on NumPy and PyTorch
        """

        super(IdxDataset, self).__init__()

        try:  # OPEN THE IDX1 FILE
            with open(file=idx1_filepath, mode="rb") as fp:
                idx1_ubytes: NDArray[np.uint8] = np.fromfile(fp, dtype=np.uint8)
        except FileNotFoundError as fnf_error:
            raise RuntimeError(f"{idx1_filepath} is not found on this computer!") from fnf_error

        try:  # OPEN THE IDX3 FILE
            with open(file=idx3_filepath, mode="rb") as fp:
                idx3_ubytes: NDArray[np.uint8] = np.fromfile(fp, dtype=np.uint8)
        except FileNotFoundError as fnf_error:
            raise RuntimeError(f"{idx3_filepath} is not found on this computer!") from fnf_error

        self.__idx1_magic: int = int.from_bytes(idx1_ubytes[:4], byteorder="big")  # idx1 magic number
        self.__idx1_count: int = int.from_bytes(idx1_ubytes[4:8], byteorder="big")  # count of the data elements (labels)

        assert (self.__idx1_count == idx1_ubytes.size - 8) and (self.__idx1_magic == 2049), (
            f"There seems to be a parsing error or the binary file {idx1_filepath} is corrupted!"
        )

        self.__labels: torch.FloatTensor = torch.FloatTensor(
            idx1_ubytes[8:].astype(np.float32)
        )  # type casting the data from np.uint8 to np.float32 since np.exp() raises FloatingPointError with np.uint8 arrays

        self.__idx3_magic: int = int.from_bytes(idx3_ubytes[:4], byteorder="big")  # idx3 magic number
        self.__idx3_count: int = int.from_bytes(idx3_ubytes[4:8], byteorder="big")  # count of the data elements (images)
        self.__image_res: tuple[int, int] = (
            int.from_bytes(idx3_ubytes[8:12], byteorder="big"),
            int.from_bytes(idx3_ubytes[12:16], byteorder="big"),
        )  # shape of each element

        assert ((self.__idx3_count * self.__image_res[0] * self.__image_res[1]) == (idx3_ubytes.size - 16)) and (
            self.__idx3_magic == 2051
        ), f"There seems to be a parsing error or the binary file {idx3_filepath} is corrupted!"

        # idx3 file stores data as bytes but we'll load in each byte as a 32 bit floats because np.exp() raises a FloatingPointError with np.uint8 type arrays
        self.__data: torch.FloatTensor = torch.FloatTensor(
            idx3_ubytes[16:].reshape(self.__idx3_count, self.__image_res[0], self.__image_res[1]).astype(np.float32)
        )

        assert self.__idx1_count == self.__idx3_count, (
            f"The pair of Idx1 and Idx3 files passed seem incompatible!, {idx3_filepath} has {self.__idx3_count} images while {idx1_filepath} has {self.__idx1_count} labels!"
        )

    @override
    def __repr__(self) -> str:
        return f"IdxDataset object (image shape: {self.__image_res}, count: {self.__idx1_count:,})"

    def __len__(self) -> int:
        return self.__idx3_count  # __idx3_count == __idx1_count, so ...

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        `Parameters`:
        index: int - the column to return from the matrix.

        `Returns`:
        (torch.FloatTensor, torch.float64) - all pixels of index th image and it's corresponding label

        `Notes`:
        IndexErrors are left to PyTorch to handle.
        """

        return self.__data[index], self.__labels[index]

    def __setitem__(self, index: int) -> None:
        raise PermissionError("IdxDataset objects are immutable!")
