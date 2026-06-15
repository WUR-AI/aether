def center_crop_npy(arr, target_shape):
    """Crops npy array to desired size."""
    if arr.shape[1] < target_shape[0]:
        raise ValueError(
            f"Requested tile size {target_shape} is larger than actual available tile size {arr.size()[1]}"
        )

    slices = []
    for dim, target in zip(arr.shape, target_shape):
        start = (dim - target) // 2
        end = start + target
        slices.append(slice(start, end))
    return arr[tuple(slices)]
