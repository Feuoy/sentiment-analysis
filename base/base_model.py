import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    模型基类，rnn和rnn+att用，继承nn.Module

    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        前向传播
        Forward pass logic

        :return: Model output
        """

        # 抛出未实现Error
        raise NotImplementedError

    def __str__(self):
        """
        打印模型，和可训练参数量

        Model prints with number of trainable parameters
        """

        # model_parameters，可训练参数的iterator（Parameter类型->Tensor类型）
        """
        requires_grad，是否求梯度
        filter()，所有self.parameters()迭代传入lambda过滤
        filter(function or None, iterable) --> filter object
        Return an iterator yielding those items of iterable for which function(item)
        is true. If function is None, return the items that are true.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())

        # params，可训练参数量
        """
        所有可训练参数的，所有维度乘积，的和
        p.size()，返回类型可能是Tensor的Size，不一定是int
        np.prod()，求内含array的乘积
        """
        params = sum([
            np.prod(p.size())
            for p in model_parameters
        ])

        # 父类（模型），子类（可训练参数量）
        """
        2020-05-29 11:54:12,920 - train - INFO - RnnModel(
          (embedding): Embedding(89197, 300)
          (rnn): LSTM(300, 256, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
          (fc): Linear(in_features=512, out_features=1, bias=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
        Trainable parameters: 29479357
        """
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
