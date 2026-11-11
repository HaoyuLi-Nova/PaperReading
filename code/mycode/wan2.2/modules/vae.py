import torch
import torch.nn as nn

class DoubleConv3d(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(),
            nn.Conv3d(cin, cout, 3,padding=1,bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLu(),
        )
    def forward(self, x):
        return self.net(x)

class Unet3D(nn.Module):
    def __init__(self, in_ch =1, out_ch=2, base=16):
        super().__init__()
        self.enc1 = DoubleConv3d(in_ch, base)
        self.enc2 = DoubleConv3d(base, base*2)
        self.bott = DoubleConv3d(base*2, base*4)
        self.pool = nn.MaxPool3d(2)
        self.up2  = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv3d(base*4, base*2)
        self.up1  = nn.ConvTranspose3d(base*2, base,   2, stride=2)
        self.dec1 = DoubleConv3d(base*2, base)
        self.head = nn.Conv3d(base, out_ch, 1)
    
    def fprward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2),e1], dim=1))
        return self.head(d1)


model = Unet3D()

x = torch.randn(1,1,32,64,64)
y = model(x)

print(y.shape)




