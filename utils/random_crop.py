import torch
import random



class RandomCrop:
    def __init__(self, points = 720):
        self.points = points

    def __call__(self, x: torch.tensor, *args):
        # print(x.shape)
        x = x.permute(0,2,1,3)
        x = x.contiguous().view(x.shape[0],x.shape[1],-1)
        # print(x.shape)
        # print(self.points)
        N, C, T = x.shape
        # print(T)
        # s = random.randint(0, T - self.points)
        s = 0
        return x[..., s:s + self.points]