import numpy as np

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3)
    
    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j
    
    def forward(self, image):
        h, w = image.shape
        output = np.zeros(h - 2, w - 2, self.num_filters)

        for img, i, j in self.iterate_regions(image):
            output[i, j] = np.sum(img * self.filters, axis=(1 , 2))
        return output