import torchvision


def embed_image(image, model):
    """Embeds an image into a vector space using a pretrained model.

    Args:
        image (torch.Tensor): Image tensor of shape (1, 3, H, W).
        model (torchvision.models): Pretrained model.

    Returns:
        torch.Tensor: Vector embedding of shape (1, D).
    """
    pass