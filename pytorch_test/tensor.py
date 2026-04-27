import torch


def basic_create_tensor():
    print(f"{'=' * 20} Creating Tensors {'=' * 20}")

    zeros = torch.zeros((2, 3), dtype=torch.int32)
    print("zeros:")
    print(zeros)

    ones = torch.ones((2, 3))
    print("\nones:")
    print(ones)

    rand = torch.rand((2, 3))
    print(torch.is_tensor(rand))
    print("\nrand:")
    print(rand)
    print(f"Shape: {rand.shape}")
    print(f"Dtype: {rand.dtype}")
    print(f"Device: {rand.device}")
    print(f"Requires Grad: {rand.requires_grad}")
    print(f"Is CUDA: {rand.is_cuda}")

    ones_like = torch.ones_like(rand)
    print("\nones (ones_like rand):")
    print(ones_like)

    full = torch.full((2, 2), 7)
    print("\nfull (full of 7):")
    print(full)

    eye = torch.eye(5)
    print("\neye (5x5 identity matrix):")
    print(eye)

    reshape_1 = torch.arange(12).reshape((3, 4))
    print("\nreshape (arange(12) reshaped to 3x4):")
    print(reshape_1)

    unsqueezed = torch.unsqueeze(reshape_1, dim=0)
    print("\nUnsqueezed tensor (dim=0):")
    print(unsqueezed)
    print(f"Shape of unsqueezed tensor: {unsqueezed.shape}")

    squeezed = torch.squeeze(reshape_1)
    print("\nSqueezed tensor:")
    print(squeezed)
    print(f"Shape of squeezed tensor: {squeezed.shape}")

    transpored = reshape_1.t()
    print("\nTransposed tensor:")
    print(transpored)


def split_and_concat_tensor():
    print(f"{'=' * 20} Splitting and Concatenating Tensors {'=' * 20}")

    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Original tensor:")
    print(a)

    b = torch.tensor([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
    print("\nAnother tensor:")
    print(b)

    # unbind the tensors
    unbind_a = torch.unbind(a, dim=0)
    print("\nUnbound tensor a:")
    for i, unbound in enumerate(unbind_a):
        print(f"Unbound {i}:")
        print(unbound)

    # Split the tensor into 3 chunks along the first dimension
    chunks = torch.chunk(a, 3, dim=0)
    print("\nChunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:")
        print(chunk)

    # Concatenate the chunks back together
    concatenated = torch.cat(chunks, dim=0)
    print("\nConcatenated tensor:")
    print(concatenated)

    hstacked = torch.hstack((a, b))
    print("\nHorizontally stacked tensor:")
    print(hstacked)

    vstacked = torch.vstack((a, b))
    print("\nVertically stacked tensor:")
    print(vstacked)

    stacked_0 = torch.stack((a, b), dim=0)
    print("\nStacked tensor (dim=0):")
    print(stacked_0)
    print(f"Shape of stacked tensor: {stacked_0.shape}")

    stacked_1 = torch.stack((a, b), dim=1)
    print("\nStacked tensor (dim=1):")
    print(stacked_1)
    print(f"Shape of stacked tensor: {stacked_1.shape}")

    stacked_2 = torch.stack((a, b), dim=2)
    print("\nStacked tensor (dim=2):")
    print(stacked_2)
    print(f"Shape of stacked tensor: {stacked_2.shape}")

    split_0 = torch.split(a, 1, dim=0)
    print("\nSplit tensor (split size=1, dim=0):")
    for i, split in enumerate(split_0):
        print(f"Split {i}:")
        print(split)

    split_1 = torch.split(a, 2, dim=1)
    print("\nSplit tensor (split size=2, dim=1):")
    for i, split in enumerate(split_1):
        print(f"Split {i}:")
        print(split)


def repeat_and_expand_tensor():
    # note" repeat and tile will copy the data, so they can be memory intensive for large tensors
    print(f"{'=' * 20} Repeating and Expanding Tensors {'=' * 20}")

    a = torch.tensor([[1, 2], [3, 4]])
    print("Original tensor:")
    print(a)

    repeated = a.repeat(2, 3)
    print("\nRepeated tensor (repeat(2, 3)):")
    print(repeated)

    tiled = a.tile((2, 3))
    print("\nTiled tensor (tile((2, 3))):")
    print(tiled)

    b = torch.tensor([5, 6])
    print("\nAnother tensor:")
    print(b)
    print("\nShape of b:")
    print(b.shape)
    # expand will not copy the data, and it can only be used to expand dimensions of size 1
    expanded = b.unsqueeze(1).expand(2, 3)
    print("\nExpanded tensor:")
    print(expanded)


def basic_operations():
    print(f"{'=' * 20} Basic Operations on Tensors {'=' * 20}")

    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])

    print("Tensor a:")
    print(a)

    print("\nTensor b:")
    print(b)

    # Element-wise addition
    add = a + b
    print("\na + b:")
    print(add)

    # Element-wise multiplication
    mul = a * b
    print("\na * b:")
    print(mul)

    # Matrix multiplication
    matmul = torch.matmul(a, b)
    print("\na @ b (matrix multiplication):")
    print(matmul)

    dot = torch.dot(a.flatten(), b.flatten())
    print("\nDot product of a and b:")
    print(dot)


def index_tensor():
    print(f"{'=' * 20} Indexing Tensors {'=' * 20}")

    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Original tensor:")
    print(a)

    # basic indexing
    print("\na[1:3, :]:")
    print(a[1:3, :])

    # boolean indexing
    print("\na[a > 5]:")
    print(a[a > 5])

    # advanced indexing
    print("\na[[0, 2], [1, 2]]:")
    print(a[[0, 2], [1, 2]])

    # indexing with ellipsis
    print("\na[..., 1]:")
    print(a[..., 1])

    # indexing with new axis
    print("\na[:, None, :]:")
    print(a[:, None, :])

    # gathering
    indices = torch.tensor([[0, 1], [1, 2], [2, 0]])
    print("\nIndices for gathering:")
    print(indices)
    print("\na.gather(1, indices):")
    print(a.gather(1, indices))
    print("\na.gather(0, indices):")
    print(a.gather(0, indices))

    # scattering
    scatter_indices = torch.tensor([[0, 1], [1, 0], [2, 2]])
    scatter_values = torch.tensor([[10, 20], [30, 40], [50, 60]])
    print("\nIndices for scattering:")
    print(scatter_indices)
    print("\nValues for scattering:")
    print(scatter_values)
    a.scatter_(1, scatter_indices, scatter_values)
    print("\na after scattering(1, scatter_indices, scatter_values):")
    print(a)

    # masking, mask will squeeze the dimensions of the output tensor, so the output will be 1D
    mask = torch.tensor(
        [[True, False, True], [False, True, False], [True, False, True]]
    )
    print("\nMask for masking:")
    print(mask)
    print("\na[mask]:")
    print(a[mask])

    # indexing with torch.where
    condition = a > 5
    print("\nCondition for torch.where (a > 5):")
    print(condition)
    print("\ntorch.where(condition, a, torch.zeros_like(a)):")
    print(torch.where(condition, a, torch.zeros_like(a)))


if __name__ == "__main__":
    # # create tensor
    # basic_create_tensor()

    # # split and concatenate tensor
    # split_and_concat_tensor()

    # # basic operations on tensors
    # basic_operations()

    # index tensor
    index_tensor()
