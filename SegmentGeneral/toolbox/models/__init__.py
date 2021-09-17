from toolbox.models.unet import unet
from toolbox.models.segnet import segnet
from toolbox.models.linknet import linknet
from toolbox.models.enet import ENet
from toolbox.models.FCN import FCN
from toolbox.models.FusionNet import Fusionnet
from toolbox.models.Deeplab_v3plus import DeepLabv3_plus
from toolbox.models.GCN import GCN
from toolbox.models.DFN import DFN
from toolbox.models.ExFuse import GCNFuse
from toolbox.models.mylinknet import MyResnet34
from toolbox.models.mylinknet import MyResnet50
from toolbox.models.mylinknet import MyResnet34WithAttention
from toolbox.models.mylinknet import MyResnet34WithTripletAttention
from toolbox.models.lambdaResnet import LambdaResnet
#from toolbox.models.ResAttentionNetLeakyRelu import ResAttentionNet
#from toolbox.models.ResAttentionNetRRelu import ResAttentionNet
#from toolbox.models.ResAttentionNetRelu import ResAttentionNet
#from toolbox.models.ResAttentionNetPRelu import ResAttentionNet

def get_model(cfg):
    usedModel = cfg["model"]
    print("cfg model ", usedModel)

    model = None
    if (usedModel == "ResAttentionNetLeakyRelu"):
        from toolbox.models.ResAttentionNetLeakyRelu import ResAttentionNet
    elif (usedModel == "ResAttentionNetRRelu"):
        from toolbox.models.ResAttentionNetRRelu import ResAttentionNet
    elif (usedModel == "ResAttentionNetRelu"):
        from toolbox.models.ResAttentionNetRelu import ResAttentionNet
    else:
        from toolbox.models.ResAttentionNetPRelu import ResAttentionNet

    return {
        'unet': unet,
        'segnet': segnet,
        'linknet': linknet,
        'enet': ENet,
        'fcn': FCN,
        'fusionnet': Fusionnet,
        'deeplabv3': DeepLabv3_plus,
        'gcn': GCN,
        'exfuse': GCNFuse,
        'dfn': DFN,
        'myresnet34': MyResnet34,
        'myresnet50': MyResnet50,
        'myresnet34WithAttention': MyResnet34WithAttention,
        'myresnet34WithTripletAttention': MyResnet34WithTripletAttention,
        'lambdaresnet': LambdaResnet,
        'resAttentionNet': ResAttentionNet

    }[cfg['model_name']](n_classes=cfg['n_classes'])

def get_modelMoreParams(model_name, n_classes):
    return {
        'unet': unet,
        'segnet': segnet,
        'linknet': linknet,
        'enet': ENet,
        'fcn': FCN,
        'fusionnet': Fusionnet,
        'deeplabv3': DeepLabv3_plus,
        'gcn': GCN,
        'exfuse': GCNFuse,
        'dfn': DFN,
        'myresnet34': MyResnet34,
        'myresnet50': MyResnet50,
        'myresnet34WithAttention': MyResnet34WithAttention,
        'myresnet34WithTripletAttention': MyResnet34WithTripletAttention,
        'lambdaresnet': LambdaResnet,
        'resAttentionNet': ResAttentionNet

    }[model_name](n_classes=n_classes)
