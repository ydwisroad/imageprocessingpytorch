from toolbox.models.unet import unet
from toolbox.models.segnet import segnet
from toolbox.models.linknet import linknet
from toolbox.models.enet import ENet
from toolbox.models.FCN import FCN
from toolbox.models.FusionNet import Fusionnet
from toolbox.models.Deeplab_v3plus import DeepLabv3_plus

def get_model(cfg):

    return {
        'unet': unet,
        'segnet': segnet,
        'linknet': linknet,
        'enet': ENet,
        'fcn': FCN,
        'fusionnet': Fusionnet,
        'deeplabv3': DeepLabv3_plus

    }[cfg['model_name']](n_classes=cfg['n_classes'])
