import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
import torch.nn.functional as F

class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)
            
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            #print("Device ", device) 
            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
                zz_channel = zz_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


class CoordConv1d(conv.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv1d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        #print("CoordConv2d ", input_tensor.shape)
        #out = self.addcoords(input_tensor)
        
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #print("Device ", device) 
        if torch.cuda.is_available and self.use_cuda:
            input_tensor = input_tensor.cuda()
            xx_channel = xx_channel.cuda()
            yy_channel = yy_channel.cuda()
        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)
        
        out = self.conv(out)

        return out


class CoordConv3d(conv.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=False):
        super(CoordConv3d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 3
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv3d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.coordconv = CoordConv2d(2, 32, 1, with_r=True)
        self.conv1 = nn.Conv2d(32, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64,  1, 1)
        self.conv4 = nn.Conv2d( 1,  1, 1)

    def forward(self, x):
        print("before coordconv ", x.shape)
        x = self.coordconv(x)
        print("after coordconv ", x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.view(-1, 64*64)
        return x

#https://github.com/walsvid/CoordConv
#https://blog.csdn.net/weixin_39553705/article/details/111136917?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-7&spm=1001.2101.3001.4242
if __name__ == '__main__':
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    testx = torch.randn(4, 2, 160, 160)
    print("input shape ", testx.shape)
    output = net(testx)
    print("output shape ",output.shape)
