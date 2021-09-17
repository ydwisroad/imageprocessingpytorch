import torch
import torch.nn as nn
from collections import OrderedDict
from models.mydarknet import darknet53
import yaml  # for torch hub


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        #add spp here

        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),

        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                        stride=1, padding=0, bias=True)
    ])
    return m

class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config
        with open(cfg) as f:
            self.yaml = yaml.load(config, Loader=yaml.FullLoader)  # model dict
        self.nc = self.yaml['nc']
        self.anchors = self.yaml['anchors']
        self.na = (len(self.anchors[0]) // 2) if isinstance(self.anchors, list) else self.anchors  # number of anchors
        self.nl = len(self.anchors)

        self.stride = torch.tensor([8, 16, 32])
        self.anchors /= self.stride.view(-1, 1, 1)

        #  backbone
        self.backbone = darknet53(None)

        out_filters = self.backbone.layers_out_filters
        #  last_layer0
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0)

        #  embedding1
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)

        #  embedding2
        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)


    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                #print("i ", i)
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch
        #  backbone
        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.last_layer0, x0)

        #  yolo branch 1
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.last_layer1, x1_in)

        #  yolo branch 2
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, _ = _branch(self.last_layer2, x2_in)
        return out0, out1, out2

class LastLayer(nn.Module):
    def __init__(self, filters_list, in_filters, out_filter):
        super(LastLayer, self).__init__()
        #pad = (kernel_size - 1) // 2 if kernel_size else 0

        self.block1 = nn.Sequential(nn.Conv2d(in_filters, filters_list[0],kernel_size=1,stride=1, padding=(1 - 1) // 2 if 1 else 0, bias=False),\
                      nn.BatchNorm2d(filters_list[0]), nn.LeakyReLU(0.1),
        nn.Conv2d(filters_list[0], filters_list[1],kernel_size=3,stride=1, padding=(3 - 1) // 2 if 3 else 0, bias=False),
        nn.BatchNorm2d(filters_list[1]),
        nn.LeakyReLU(0.1),
        nn.Conv2d(filters_list[1], filters_list[0],kernel_size=1,stride=1, padding=(1 - 1) // 2 if 1 else 0, bias=False),
        nn.BatchNorm2d(filters_list[0]),
        nn.LeakyReLU(0.1))

        #add SPP here
        k =5
        stride = 1
        self.maxPool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=(5 - 1) // 2)
        k = 9
        self.maxPool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=(9 - 1) // 2)
        k = 13
        self.maxPool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=(13 - 1) // 2)

        self.blockSppAdditional = nn.Sequential(nn.Conv2d(4 * filters_list[0], filters_list[1],kernel_size=3,stride=1, padding=(3 - 1) // 2 if 3 else 0, bias=False),
                                    nn.BatchNorm2d(filters_list[1]),nn.LeakyReLU(0.1),\
                      nn.Conv2d(filters_list[1], filters_list[0],kernel_size=1,stride=1, padding=(1 - 1) // 2 if 1 else 0, bias=False),\
                      nn.BatchNorm2d(filters_list[0]),nn.LeakyReLU(0.1))

        self.block2 = nn.Sequential(nn.Conv2d(filters_list[0], filters_list[1],kernel_size=3,stride=1, padding=(3 - 1) // 2 if 3 else 0, bias=False),
                                    nn.BatchNorm2d(filters_list[1]),nn.LeakyReLU(0.1),\
                      nn.Conv2d(filters_list[1], filters_list[0],kernel_size=1,stride=1, padding=(1 - 1) // 2 if 1 else 0, bias=False),\
                      nn.BatchNorm2d(filters_list[0]),nn.LeakyReLU(0.1))


        self.block3 = nn.Sequential(nn.Conv2d(filters_list[0], filters_list[1], kernel_size=3, stride=1, padding=(3 - 1) // 2 if 3 else 0, bias=False),\
                      nn.BatchNorm2d(filters_list[1]),nn.LeakyReLU(0.1),\
                      nn.Conv2d(filters_list[1], out_filter, kernel_size=1,stride=1, padding=0, bias=True))

    def forward(self, x):
        xIn = x
        x  = self.block1(xIn)
        #Add SPP Here  https://blog.csdn.net/qq_33270279/article/details/103898245
        print("x size ", x.size())
        branch1 = self.maxPool1(x)
        print("branch1 size ", branch1.size())
        branch2 = self.maxPool2(x)
        print("branch2 size ", branch2.size())
        branch3 = self.maxPool3(x)
        print("branch3 size ", branch3.size())
        sppOut = torch.cat((x, branch1, branch2, branch3),1)
        print("sppOut ", sppOut.size())

        sppX = self.blockSppAdditional(sppOut)
        print("sppX ", sppX.size())

        #If spp
        xOutBranch = self.block2(sppX)
        #else
        #xOutBranch = self.block2(x)

        x = self.block3(xOutBranch)

        return x, xOutBranch


class YoloBodyNew(nn.Module):
    def __init__(self, yamlFile, config):
        super(YoloBodyNew, self).__init__()

        with open(yamlFile) as f:
            self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        print("got yaml config ", self.yaml)
        self.nc = self.yaml['nc']
        self.no = self.nc + 5  # number of outputs per anchor
        self.anchors = config["yolo"]["anchors"]
        print("initial anchors ", self.anchors)
        self.na = len(self.anchors[0])
        self.nl = len(self.anchors)

        self.stride = torch.tensor([8, 16, 32])
        strideView = self.stride.view(-1, 1, 1)
        print("stride view ", strideView)
        self.anchors = torch.tensor(self.anchors).float()/ strideView.float()
        print("self anchors after divide ", self.anchors)
        #  backbone
        self.backbone = darknet53(None)

        out_filters = self.backbone.layers_out_filters
        #  last_layer0
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.last_layer0 = LastLayer([512, 1024], out_filters[-1], final_out_filter0)

        #  embedding1
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = LastLayer([256, 512], out_filters[-2] + 256, final_out_filter1)

        #  embedding2
        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = LastLayer([128, 256], out_filters[-3] + 128, final_out_filter2)

        self.stride = torch.tensor([ 8., 16., 32.])

        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(self.anchors).float()   #.view(self.nl, -1, 2)
        print("a size ", a.size())
        #self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)

    def forward(self, x, augment=False):
        #  backbone
        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        #print("x0 size ", x0.size())
        out0, out0_branch = self.last_layer0(x0)
        #print("out0 size ", out0.size() , " ", out0_branch.size())
        #  yolo branch 1
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = self.last_layer1(x1_in)

        #  yolo branch 2
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, _ = self.last_layer2(x2_in)

        # Decoder to be added
        #out1 = out1.view(batch_size, self.num_anchors,
        #self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
        #bs, _, ny, nx = out2.shape  # x(bs,678,20,20) to x(bs,3,20,20,226)
        #out2 = out2.view(bs, -1, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        #bs, _, ny, nx = out1.shape  # x(bs,678,20,20) to x(bs,3,20,20,226)
        #out1 = out1.view(bs, -1, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        #bs, _, ny, nx = out0.shape  # x(bs,678,20,20) to x(bs,3,20,20,226)
        #out0 = out0.view(bs, -1, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        out = []
        out.append(out2)
        out.append(out1)
        out.append(out0)
        testx = out.copy()

        z = []  # inference output
        for i in range(self.nl):
            testx[i] = out[i]
            bs, _, ny, nx = testx[i].shape     # x(bs,255,20,20) to x(bs,3,20,20,85)
            testx[i] = testx[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != testx[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(testx[i].device)

                y = testx[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(testx[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return testx if self.training else (torch.cat(z, 1), testx)  #out2, out1, out0

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

if __name__ == "__main__":
    #Yolo3 Test
    print("Test if yolo3")