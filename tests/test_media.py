import pytest
from PIL import Image, ImageDraw
from rts.io.media import generate_image_pyramid


def generate_test_image(width: int, height: int) -> Image:
    """Generates a blank white image of the given dimensions."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "test", fill="black")
    return image


def test_generate_image_pyramid():
    # Create a test image of 512x512 pixels
    test_image = generate_test_image(512, 512)

    # Generate pyramid
    files = generate_image_pyramid(test_image, "test", base_res=16, depth=6)

    assert len(files) == 6

    # Verify the resolution of each image and the 'key' structure
    for key in files:
        files[key].seek(0)  # Reset the BytesIO position
        img = Image.open(files[key])

        if 'original' in key:
            assert img.size == (512, 512)
            assert key.split("/")[1] == "test.jpg"
        else:
            size = int(key.split("/")[0].split('px')[0])
            assert img.size == (size, size)
            assert key == f"{size}px/test.jpg"
