import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial


class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """

    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()

        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_timesteps).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        """
        param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        """
        # (X·W1)·W2,  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        part1 = torch.matmul(torch.matmul(x, self.W1), self.W2)
        # (W3·X)^T, # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        part2 = torch.matmul(self.W3, x).transpose(-1, -2)
        # (b,N,T)(b,T,N) -> (B, N, N)
        product = torch.matmul(part1, part2)
        # (N,N)(B, N, N)->(B,N,N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAT(nn.Module):
    """
    K-order chebyshev graph convolution
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super(cheb_conv_withSAT, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))
                                       for _ in range(K)])

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :param spatial_attention
        :return: (batch_size, N, F_out, T)
        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)  # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in)
                # 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()

        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        """
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)
        part1 = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # (F)(B,N,F,T)->(B, N, T)
        part2 = torch.matmul(self.U3, x)
        product = torch.matmul(part1, part2)
        # (B, T, T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        E_normalized = F.softmax(E, dim=1)
        return E_normalized


class cheb_conv(nn.Module):
    """
    K-order chebyshev graph convolution， 与cheb_conv_withSAT相比，该函数没有使用注意力机制
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        :param K: int
        :param in_channels: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        """
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))
                                       for _ in range(K)])

    def forward(self, x):
        """
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        """

        Parameters
        ----------
        DEVICE
        in_channels
        K
        nb_chev_filter: 切比雪夫模块filter的数目
        nb_time_filter：时间模块的filter的数目
        time_strides
        cheb_polynomials：节点的邻接矩阵用于多项式
        num_of_vertices
        num_of_timesteps
        """
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAT(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.layer_normalize = nn.LayerNorm(nb_time_filter)     # 需要将channel放到最后一个纬度上

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # TAt
        temporal_At = self.TAt(x)   # (b, T, T)  b-> batch_size

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At)\
            .reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # SAt
        spatial_At = self.SAt(x_TAt)

        # cheb_gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)     # (b, N, F, T)

        # convolution along the time axis, # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))

        # residual shortcut, # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))

        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)
        output = self.layer_normalize(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return output


class ASTGCN_submodule(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                 num_for_predict, len_input, num_of_vertices):
        super(ASTGCN_submodule, self).__init__()

        self.block_list = nn.ModuleList([ASTGCN_block(DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                                                      time_strides, cheb_polynomials, num_of_vertices, len_input)])
        self.block_list.extend([ASTGCN_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1,
                                             cheb_polynomials, num_of_vertices, len_input//time_strides)
                                for _ in range(nb_block-1)])
        self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self.DEVICE = DEVICE
        self.to(DEVICE)

    def forward(self, x):
        """
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        """
        for block in self.block_list:
            x = block(x)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)

        return output


def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx, num_for_predict,
               len_input, num_of_vertices):

    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = ASTGCN_submodule(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                             cheb_polynomials, num_for_predict, len_input, num_of_vertices)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
