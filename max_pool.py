import torch
import torch.nn as nn

class MaxPool2D:
    def __init__(self):
        pass
    
    def generate_regions(self, image):
        h, w, _ = image.shape
        new_h, new_w = h // 2, w // 2
        for i in range(new_h):
            for j in range(new_w):
                img = image[2 * i : (2 * i + 2), 2 * j : (2 * j + 2)]
                yield img, i, j

    def forward(self, x):
        h, w, num_filters = x.shape
        output = torch.zeros((h // 2, w // 2, num_filters))

        for img, x, y in self.generate_regions(x):
            output[x, y] = torch.max(img, axis=(0, 1))

        return output