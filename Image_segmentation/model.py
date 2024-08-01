import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        print("Size of x in doubleconv", x.shape)
        print("hello downs2")
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.downs2 = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # sub Down part of UNET
        for feature in features:
            self.downs2.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # print("Shape of x before down: ", x.shape)
        for down, down2 in zip(self.downs, self.downs2):
            # print("Shape of x before down(x): ", x.shape)
            print(x.shape)
            x = down(x)
            #print(x1)
            #print(x.shape)
            x2 = down2(x)
            #x = torch.cat((x1, x2), dim=1)
            # print("Shape of x before skip-connnection: ", x.shape)
            skip_connections.append(x)
            # print("Shape of x after skip-conn: ", x.shape)
            x = self.pool(x)
            # print("Shape of x after pooling: ", x.shape)
        # print("Shape of x after down: ", x.shape)  #shape has to be [16, 1024, 10, 15]

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        #print("Skip_connection list", skip_connections)
        # print("Shape of x after bottleneck: ", x.shape)   #shape has to be 2048

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            #print("Shape of x before concat: ", x.shape)
            #print("Shape of concat var:", skip_connection.shape)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            #print("Shape after contentation:", concat_skip.shape)

            x = self.ups[idx+1](concat_skip)
        # print("Shape of x after ups: ", x.shape)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()