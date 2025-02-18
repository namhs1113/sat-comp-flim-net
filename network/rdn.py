
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U


class RDB_conv(nn.Module):
    def __init__(self, in_channels, grow_rate, act=nn.ReLU(True), k_size=3):
        super(RDB_conv, self).__init__()

        Cin = in_channels
        G = grow_rate

        conv = list()
        conv.append(nn.Conv2d(in_channels=Cin, out_channels=G,
                              kernel_size=k_size, padding=1, padding_mode="replicate", stride=1))
        conv.append(act)
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        out = self.conv(x)
        return out


class RDB(nn.Module):
    def __init__(self, grow_rate0, grow_rate, n_conv_layer, act=nn.ReLU(True), k_size=3):
        super(RDB, self).__init__()

        G0 = grow_rate0
        G = grow_rate
        C = n_conv_layer

        convs = list()
        for c in range(C):
            convs.append(RDB_conv(G0 + c*G, G, act, k_size))
        self.convs = nn.Sequential(*convs)

        # Local feature fusion (1x1 conv)
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        x0 = x
        mem = [x,]
        for conv in self.convs:
            in_ = torch.cat(mem, dim=1)
            x = conv(in_)
            mem.append(x)

        return self.LFF(torch.cat(mem, dim=1)) + x0


class RDN(nn.Module):
    def __init__(self, **kwargs):
        super(RDN, self).__init__()
        G0 = kwargs["G0"]
        act = kwargs["act"]
        k_size = kwargs["k_size"]

        # number of RDB blocks, conv layers, out channels
        D, C, G = kwargs["D"], kwargs["C"], kwargs["G"]

        # Shalldow feature extraction net
        self.SFENet1 = nn.Conv2d(kwargs["n_in_ch"], G0, k_size, padding=1, padding_mode="replicate", stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, k_size, padding=1, padding_mode="replicate", stride=1)

        # Residual dense blocks and dense feature fusion
        RDBs = list()
        for i in range(D):
            RDBs.append(RDB(grow_rate0=G0, grow_rate=G, n_conv_layer=C, act=act, k_size=k_size))
        self.RDBs = nn.Sequential(*RDBs)

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(D*G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, k_size, padding=1, padding_mode="replicate", stride=1)
        ])

        # Final layer (instead of upsampling layer)
        self.final = nn.Sequential(*[
            nn.Conv2d(G0, G, k_size, padding=1, padding_mode="replicate", stride=1),
            nn.Conv2d(G, kwargs["n_out_ch"], k_size, padding=1, padding_mode="replicate", stride=1)
        ])

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i, RDB in enumerate(self.RDBs):
            x = RDB(x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, dim=1))
        x = x + f__1
        y = self.final(x)

        return y


class SN_PatchGAN_D(nn.Module):
    def __init__(self, in_channels=3, num_features=64):
        super(SN_PatchGAN_D, self).__init__()

        # Discriminator modules
        self.conv1 = U.spectral_norm(nn.Conv2d(in_channels, num_features, kernel_size=5, stride=2))
        self.conv2 = U.spectral_norm(nn.Conv2d(num_features, 2 * num_features, kernel_size=5, stride=2))
        self.conv3 = U.spectral_norm(nn.Conv2d(2 * num_features, 4 * num_features, kernel_size=5, stride=2))
        self.conv4 = U.spectral_norm(nn.Conv2d(4 * num_features, 4 * num_features, kernel_size=5, stride=2))
        self.conv5 = U.spectral_norm(nn.Conv2d(4 * num_features, 4 * num_features, kernel_size=5, stride=2))

    def forward(self, x, leaky_alpha=0.2):
        out = F.leaky_relu(self.conv1(x), leaky_alpha)
        out = F.leaky_relu(self.conv2(out), leaky_alpha)
        out = F.leaky_relu(self.conv3(out), leaky_alpha)
        out = F.leaky_relu(self.conv4(out), leaky_alpha)
        out = F.leaky_relu(self.conv5(out), leaky_alpha)

        return out