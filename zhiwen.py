import os
import sys
import numpy as np
from pdb import set_trace as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style


'''
https://github.dev/vzhou842/cnn-from-scratch
'''
class Conv3x3():
    def __init__(self, num_filters) -> None:
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array.
        '''
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j

    def forward(self, x):
        '''
        '''
        self.last_input = x
        h, w = x.shape
        output = np.zeros([h - 2, w - 2, self.num_filters])

        for im_region, i, j in self.iterate_regions(x):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backward(self, d_L_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        self.filters -= learn_rate * d_L_d_filters
        # We aren't returning anything here since we use Conv3x3 as the first layer in our CNN.
        # Otherwise, we'd need to return the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None

class MaxPool2d:

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''
        h, w, _ = image.shape
        new_h, new_w = h // 2, w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(2*i):(2*i+2), (2*j):(2*j+2)]
                yield im_region, i, j

    def forward(self, x):
        self.last_input = x
        h, w, num_filters = x.shape
        output = np.zeros([h // 2, w // 2, num_filters])
        for im_regios, i, j in self.iterate_regions(x):
            output[i, j] = np.amax(im_regios, axis=(0, 1))
        return output

    def backward(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]
        return d_L_d_input


class Softmax:
    def __init__(self, input_len, nodes):
        '''
        '''
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)


'''Code for linear regression gradient decent
'''
class LinearRegression:

    def __init__(self):
        # https://blog.csdn.net/tsinghuahui/article/details/80223039
        # Create an attribute to log the loss
        self.loss = []

        # define other params
        self.n_iterations = 1000
        self.data_n = 10000
        x1 = np.random.randn(self.data_n)
        x2 = np.random.randn(self.data_n)
        x3 = np.random.randn(self.data_n)
        noise = np.random.normal(0, 5, self.data_n)
        # Initialize parameters
        self.y = 100*x1 - 50*x2 + 25*x3 + 10 + noise
        self.X = np.array([x1,x2,x3]).T
        self.bias = 0


    def fit(self, alpha = 0.05, n_iterations = 100):
        X = self.X
        y = self.y
        # Get num observations and num features
        self.n, self.m = X.shape

        # Create array of weights, one for each feature
        self.weights = np.ones(self.m)
        print(f'> weights init: {self.weights}')

        # Iterate a number of times
        for _ in range(self.n_iterations):

            # Generate prediction
            y_hat = np.dot(X, self.weights) + self.bias

            # Calculate error
            error = y - y_hat

            # Calculate loss (mse)
            mse = np.square(error).mean()

            # Log the loss
            self.loss.append(mse)

            # Calculate gradients using partial derivatives
            gradient_wrt_weights = - (1 / self.n) * np.dot(X.T, error)
            gradient_wrt_bias = - (1 / self.n) * np.sum(error)

            # Update parameters using gradients and alpha
            self.weights = self.weights - alpha * gradient_wrt_weights
            self.bias = self.bias - alpha * gradient_wrt_bias

    def predict(self):
        # Generate predictions using current weights and bias
        ret = np.dot(self.X, self.weights) + self.bias
        print(f'> weights pred: {self.weights}')
        print(f'> weights gt: 100, 50, 25, 10')
        return


'''Code for MLP
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, dim_in=3, dim=256, dim_out=2, nn_depth=8):
        super().__init__() # be aware of this line
        self.dim_in = dim_in
        self.dim = dim
        self.dim_out = dim_out
        self.nn_depth = nn_depth
        self.linears = nn.ModuleList(
            [nn.Linear(self.dim_in, self.dim)] +
            [nn.Linear(self.dim, self.dim) for i in range(self.nn_depth - 2)] +
            [nn.Linear(self.dim, self.dim_out)]
        )

    def forward(self, x):
        for idx, _ in enumerate(range(self.nn_depth)):
            x = self.linears[idx](x)
        return x



'''Code for torch rotate image
ref: https://stackoverflow.com/questions/64197754/how-do-i-rotate-a-pytorch-image-tensor-around-its-center-in-a-way-that-supports
'''
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class TorchRot():
    def __init__(self):
        img = cv2.imread('/ssd1/zhiwen/projects/code/files/lasmeninas0.png')
        self.img_t = torch.from_numpy(img).float()
        self.img_t = self.img_t.unsqueeze(0)
        self.img_t = self.img_t.permute(0, 3, 1, 2)
        print(f'> input shape:', self.img_t.shape)
        self.dtype = self.img_t.dtype
        self.device = self.img_t.device

    def rot_img(self, theta):
        rot_mat = self.get_rot_mat(theta)[None, ...].type(self.dtype).repeat(self.img_t.shape[0], 1, 1) # B x 2 x 3
        grid = F.affine_grid(rot_mat, self.img_t.size()).type(self.dtype) # B x C x H x W
        x = F.grid_sample(self.img_t, grid)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_rot_mat(self, theta):
        theta = torch.Tensor([theta])
        return torch.Tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]])



def matrix_mul(A, B):
    '''Complexity analysis
    https://stackoverflow.com/questions/3744094/time-and-space-complexity-of-vector-dot-product-computation
    '''
    row_A = len(A)
    col_A = len(A[0])
    row_B = len(B)
    col_B = len(B[0])
    AB = [[0] * col_B for _ in range(row_A)]
    for i in range(row_A):
        for j in range(col_B):
            for k in range(col_A):
                AB[i][j] += A[i][k] * B[k][j]
    return AB

class SparseVector():
    def __init__(self, nums):
        self.dic = {}
        for i, n in enumerate(nums):
            if n != 0:
                self.dic[i] = n

    def dotProduct(self, vec):
        res = 0
        if len(self.dic) > len(vec.dic):
            for j, n in vec.dic.items():
                if j in self.dic:
                    res += n * self.dic[j]
        else:
            for i, n in self.dic.items():
                if i in vec.dic:
                    res += n * vec.dic[i]
        return res

import numpy as np
class ComputeMeanVar():
    def __init__(self):
        self.data = [21, 22, 23]

    def get_mean(self):
        return np.mean(self.data)

    def get_var(self):
        return np.var(self.data)


class MatrixTranspose():
    def __init__(self):
        self.matrix = [[1,2,3],[4,5,6],[7,8,9]]

    def transpose(self):
        ret = []
        col = len(self.matrix[0])
        for i in range(col):
            ret.append([row[i] for row in self.matrix])
        return ret

class FindSubmatrixSum():
    '''We need create a sum matrix sum[][] which has the same shape with matrix
    '''
    def __init__(self):
        self.matrix = [
            [0, 2, 5, 4, 1],
            [4, 8, 2, 3, 7],
            [6, 3, 4, 6, 2],
            [7, 3, 1, 8, 3],
            [1, 5, 7, 9, 4]
        ]
        self.p = self.q = 1
        self.r = self.s = 3

    def find_submatrix(self):
        M, N = len(self.matrix), len(self.matrix[0])
        # create sum matrix
        s = [[0 for x in range(len(self.matrix[0]))] for y in range(len(self.matrix))]
        s[0][0] = self.matrix[0][0]

        # compute the sum matrix of the first row
        for j in range(1, len(self.matrix[0])):
            s[0][j] = self.matrix[0][j] + s[0][j-1]

        # compute the sum matrix of the first column !!!!
        for i in range(1, len(self.matrix)):
            s[i][0] = self.matrix[i][0] + s[i-1][0]

        # compute the sum matrix of the rest
        for i in range(1, len(self.matrix)):
            for j in range(1, len(self.matrix[0])):
                s[i][j] = self.matrix[i][j] + s[i-1][j] + s[i][j-1] - s[i-1][j-1]

        total = s[self.r][self.s]
        # find the sum of the submatrix
        if self.q - 1 >= 0:
            total -= s[self.r][self.q-1]

        if self.p - 1 >= 0:
            total -= s[self.p-1][self.s]

        if self.p - 1 >= 0 and self.q - 1 >= 0:
            total += s[self.p-1][self.q-1]

        return total


import math
import torch
import numpy as np
class ComputeGrad():
    def __init__(self):
        # a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
        # b = torch.sin(a)
        # c = 2 * b
        # d = c + 1
        # out = d.sum()
        # print(d.grad_fn)
        # print(d.grad_fn.next_functions)
        # print(d.grad_fn.next_functions[0][0].next_functions)
        # print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
        # print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
        x = torch.tensor(np.pi, requires_grad = True)
        # y = 3 * x ** 2
        y = 3 * torch.sin(x)
        y.backward() # .backward: Computes the gradient of current tensor w.r.t. graph leaves
        print("Dervative of the equation at x = 3 is: ", x.grad)


class KMeans():
    '''Reference:https://gist.github.com/colonialjelly/e980d83cae2f909c1cb777c697aa0369
    '''
    def __init__(self):
        self.X = np.random.randn(1000, 2)
        self.k = 4

    def k_means(self, centers=None, num_iter=100):
        if centers is None:
            rnd_centers_idx = np.random.choice(np.arange(self.X.shape[0]), self.k, replace=False)
            centers = self.X[rnd_centers_idx]
        for _ in range(num_iter):
            distances = np.sum(np.sqrt((self.X - centers[:, np.newaxis]) ** 2), axis=-1)
            cluster_assignments = np.argmin(distances, axis=0)
            for i in range(self.k):
                msk = (cluster_assignments == i)
                centers[i] = np.mean(self.X[msk], axis=0) if np.any(msk) else centers[i]

        return cluster_assignments, centers


class Attention(nn.Module):
    def __init__(self, dim, num_head=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        assert dim % num_head == 0, 'dim should be divided by num_head'
        self.num_head = num_head
        head_dim = dim // num_head
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4) # [3, B, numb_head, N, C//numb_head]
        q, k, v = qkv.unbind(0) # each is [B, numb_head, N, C//numb_head]

        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, numb_head, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # [B, numb_head, N, N] @ [B, numb_head, N, C//numb_head] -> [B, numb_head, N, C//numb_head] -> [B, N, numb_head,C//numb_head] -> [B, N, C]
        x = self.proj(x) # another linear!
        x = self.proj_drop(x)
        return x


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
class ComputePrcRec():
    def __init__(self, n_data):
        self.pred = (np.random.rand(n_data) > 0.5).astype(np.uint8)
        self.gt = (np.random.rand(n_data) > 0.5).astype(np.uint8)
        print(f'data init > pred: {self.pred}', )
        print(f'data init > gt:   {self.gt}', )

    def compute_score(self):
        '''
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        '''
        precision = precision_score(self.gt, self.pred)
        recall = recall_score(self.gt, self.pred)
        print(f' > precision: {precision}', )
        print(f' > recall   :   {recall}', )


class ResNetBlock(nn.Module):
    super().__init__()
    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = self.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def conv3x3(self, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
        """3x3 convolution with padding"""
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


if __name__ == '__main__':
    '''grad decent code
    '''
    # lr = LinearRegression()
    # lr.fit()
    # lr.predict()

    '''write a MLP
    '''
    # mlp = MLP()
    # x = torch.rand(100, 3)
    # res = mlp(x)
    # print(f'> mlp: {mlp}, {res.shape}')

    '''Pytorch Rotate
    '''
    # torch_rot = TorchRot()
    # res = torch_rot.rot_img(np.pi/2)
    # res = res.squeeze(0)
    # res = res.numpy().astype(np.uint8)
    # cv2.imwrite('files/rot_res.png', res)

    '''matrix mul
    '''
    # A = [[1,0,0],[-1,0,3]]
    # B = [[7,0,0],[0,0,0],[0,0,1]]
    # print(matrix_mul(A, B))

    '''Sparse vector dot
    '''
    # nums1 = [0,1,0,0,2,0,0]
    # nums2 = [1,0,0,0,3,0,4]
    # v1 = SparseVector(nums1)
    # v2 = SparseVector(nums2)
    # ans = v1.dotProduct(v2)
    # print(f'Input 1: {nums1}, Input 2: {nums2}, Res: {ans}')

    '''Compuate mean var
    '''
    # compute_mean_var = ComputeMeanVar()
    # print(f'> mean: {compute_mean_var.get_mean()}')
    # print(f'> var: {compute_mean_var.get_var()}')

    '''Matrix Transpose
    '''
    # mt = MatrixTranspose()
    # print(f'> before transpose', mt.matrix)
    # print(f'> after transpose', mt.transpose())

    '''find sum of submatrix
    '''
    # fs = FindSubmatrixSum()
    # res = fs.find_submatrix()
    # print('> matrix ', fs.matrix)
    # print('> p, q ', fs.p, fs.q)
    # print('> r, s ', fs.r, fs.s)
    # print('> ', res)

    '''Use autograd to compute grad
    '''
    # cg = ComputeGrad()

    '''KMeans
    '''
    # kmeans = KMeans()
    # kmeans.k_means()

    '''ViT Attention Block
    '''
    # attention_block = Attention(dim=384)
    # x = torch.ones([1, 192, 384])
    # y = attention_block(x)
    # print(f'y.shape:, {y.shape}')

    '''Compute Precision Recall
    '''
    precision_recall = ComputePrcRec(5)
    precision_recall.compute_score()