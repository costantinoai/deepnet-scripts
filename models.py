from imports import *


class SiameseNetEncoder(nn.Module):
    def __init__(self):
        super(SiameseNetEncoder, self).__init__()
        ## ENCODER
        # V1 layers
        self.V1_per = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=7 // 2),  # + self.vfb,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # V2 layers
        self.V2_per = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # V4 layers
        self.V4_per = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # IT layers
        self.IT = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # V1 layers
        self.V1_fov = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=7 // 2),  # + self.vfb,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # V2 layers
        self.V2_fov = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # V4 layers
        self.V4_fov = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # get features vector
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten()#, nn.Linear(512, 2), # 512*2 because we concat two outputs
        )


    def forward_once(self, inp):
        x = inp
        v1 = self.V1(x)
        v2 = self.V2(v1)
        v4 = self.V4(v2)
        vIT = self.IT(v4)
        ft = self.fc(vIT)
        return ft

    def forward(self, inp):
        inp1 = inp[0]
        inp2 = inp[1]

        out1 = self.forward_once(inp1)
        out2 = self.forward_once(inp2)
        # eucl_dist = F.pairwise_distance(out1, out2)

        # pred = self.decoder(torch.cat([out1, out2], dim=1))
        return cast(out1, Tensor), cast(out2, Tensor)#, cast(pred, Tensor)
