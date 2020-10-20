import torch.nn.functional as F
import torch


class LambdaLayer(torch.nn.Module):

    def __init__(self, lambd):

        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):

        return self.lambd(x)


inputs = torch.randn(1, 1, 4, 4)
planes = 9
shortcut1 = LambdaLayer(lambda x:F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
out1 = shortcut1(inputs)
print("finish")