from toolbox.models.unet import unet
from toolbox.models.segnet import segnet
from toolbox.models.linknet import linknet
from toolbox.models.enet import ENet


def get_model(cfg):

    return {
        'unet': unet,
        'segnet': segnet,
        'linknet': linknet,
        'enet': ENet,

    }[cfg['model_name']](n_classes=cfg['n_classes'])
