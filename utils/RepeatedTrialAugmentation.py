import torch


class RepeatedTrialAugmentation:
    def __init__(self, transform = torch.nn.Identity(), m = 5):
        self.transform = transform
        self.m = m

    def __call__(self, x, y):
        """
        m表示重复变换的次数
        :param x:
        :param y:
        :return:
        """
        if self.m == 1:
            return self.transform(x, y), y

        """
        重复变换m次，将结果cat起来
        """
        data = torch.cat([self.transform(x.clone(), y) for _ in range(self.m)], dim=0)
        labels = torch.cat([y]*self.m, dim=0)
        return data, labels