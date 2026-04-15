def center_crop_npy(arr, target_shape):
    """Crops npy array to desired size."""
    slices = []
    for dim, target in zip(arr.shape, target_shape):
        start = (dim - target) // 2
        end = start + target
        slices.append(slice(start, end))
    return arr[tuple(slices)]
