import warnings
import numpy as np
from typing import List, Optional
from img2vec_pytorch import Img2Vec
from PIL import Image
from pathlib import Path

import emv.utils
LOG = emv.utils.get_logger()


_model = {
    'name': None,
    'model': None
}


def load_model(model_name: str = 'resnet50') -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        global _model
        if not model_name:
            if not _model['name']:
                model_name = 'resnet50'
            else:
                model_name = _model['name']
        # not in cache
        if not _model['name']:
            LOG.info(f'Load model: {model_name}')
            _model['model'] = Img2Vec(cuda=True, model=model_name)
            _model['name'] = model_name

        # invalidate model
        if model_name != _model['name']:
            LOG.info(f'Change model: {model_name}')
            _model['model'] = Img2Vec(cuda=True, model=model_name)
            _model['name'] = model_name


def embed_images(images: List[np.ndarray], model_name: str = 'resnet50') -> Optional[np.ndarray]:
    if not images:
        return None

    global _model
    load_model(model_name=model_name)

    if isinstance(images, np.ndarray):
        images = [images]

    images = [Image.fromarray(image).convert('RGB') for image in images]
    vecs = _model['model'].get_vec(images, tensor=False)
    return vecs
