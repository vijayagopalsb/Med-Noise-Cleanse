from PIL import Image

def resize_image(image_path, size=(256, 256)):
    image = Image.open(image_path).convert("L")  # convert to grayscale
    return image.resize(size)
