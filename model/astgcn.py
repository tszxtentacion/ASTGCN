import torch
import torch.nn as nn


class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """

    def __init__(self, *args, **kwargs):
        super(Spatial_Attention_layer, self).__init__(*args, **kwargs)
        # 初始化参数
        self.W_1 = nn.Parameter(torch.Tensor(1, 1))

    def forward(self, x):
        """
        Parameters
        ----------
        x: mx.ndarray, x^{(r - 1)}_h,
           shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        S_normalized: mx.ndarray, S', spatial attention scores
                      shape is (batch_size, N, N)

        """
        # 得到输出矩阵的shape
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
