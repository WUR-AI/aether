from rasterio import open as ropen
from torchvision.transforms import v2


def open_file_as_tensor(path):
    """Opens tif path as tensor."""
    img = ropen(path).read()
    tensor = v2.ToImage()(img).permute(1, 2, 0)
    return tensor


