from toolbox.models.alexnet import AlexNet
from toolbox.models.googlenet import GoogLeNet
from toolbox.models.resnet import ResNet
from toolbox.models.vggnet import VGG


def get_model(cfg):

    return {
        'alexnet': AlexNet,
        'googlenet': GoogLeNet,
        'resnet': ResNet,
        'vgg': VGG
    }[cfg['model_name']](num_classes=cfg['num_classes'])
