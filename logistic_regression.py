import numpy as np



def forward(X, W):
    return np.matmul(X, W)

def _softmax(logits):
    return 1 / (1 + np.exp(-logits))

def loss(Y, y_hat, W, Lambda):
    error = (-1/n) * np.sum(
        Y * np.log(y_hat) +
        (1 - Y) * np.log(1 - y_hat)
    ) + np.linalg.norm(W, 2)
    return error, np.sum(np.argmax(y_hat, 1) == y)/n

def gradient(y_hat, Y, X, W):
    return (1/n) * np.matmul(X.T, (y_hat - Y)) + 2 * Lambda * W


def fit_regression(W, X, Y, alpha, Lambda, max_iter):
    for iter in range(max_iter):
        y_hat = forward(X, W)
        y_hat = _softmax(y_hat)
        _error, _loss = loss(Y, y_hat, W, Lambda)

        grad = gradient(y_hat, Y, X, W)
        W = W - alpha * grad
        if iter % 100 == 0: print(_error, _loss)


n, d, p = 1000, 8, 3
alpha, Lambda, max_iter = 0.0001, 0.01, 10000
X = np.random.random([n, d])
W = np.random.random([d, p])

y = np.random.randint(p, size=(1,n))
Y = np.zeros((n,p))
Y[np.arange(n), y] = 1

fit_regression(W, X, Y, alpha, Lambda, max_iter)