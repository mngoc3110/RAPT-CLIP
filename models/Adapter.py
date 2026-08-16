import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, ratio=0.2):
        super(Adapter, self).__init__()
        self.ratio = ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.Dropout(0.1)
        )
        # Zero-init second projection layer so at step 0 Adapter is exact identity (preserves CLIP features)
        nn.init.zeros_(self.fc[3].weight)

    def forward(self, x):
        return x + self.ratio * self.fc(x)
