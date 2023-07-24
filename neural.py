import random


class neural():
    layers: list[list[float]]
    grads: list[list[float]]
    weights: list[list[list[float]]]

    learning_rate: float

    def __init__(self, each_layer_size: list[int], *, learning_rate: float, activate_func, activate_func_d):
        self.layers = [[0] * size for size in each_layer_size]
        self.grads = [[0] * size for size in each_layer_size]
        self.weights = [[[0] * size_left for _ in range(size_right)] for size_left,
                        size_right in zip(each_layer_size[:-1], each_layer_size[1:])]
        for weight in self.weights:
            for wl in weight:
                for i in range(len(wl)):
                    wl[i] = 1 - random.random() * 2
        self.each_layer_size = each_layer_size
        self.learning_rate = learning_rate
        self.activate_func = activate_func
        self.activate_func_d = activate_func_d

    def forward(self, input_data: list[float]):
        self.layers = [[0] * size for size in self.each_layer_size]
        self.layers[0] = [*input_data]
        for layer_left, layer_right, weight in zip(self.layers, self.layers[1:], self.weights):
            for i in range(len(layer_right)):
                layer_right[i] = self.activate_func(sum(
                    w * ll for w, ll in zip(weight[i], layer_left)))

    def backward(self, ideal_output: list[float], *, loss_func_d=None):
        self.grads = [[0] * size for size in self.each_layer_size]
        self.grads[-1] = [(2 / self.each_layer_size[-1] * (real - ideal) if loss_func_d is None else loss_func_d(real, ideal)) for real,
                          ideal in zip(self.layers[-1], ideal_output)]
        for layer_left, layer_right, grad_left, grad_right, weight in zip(self.layers[::-1][1:], self.layers[::-1], self.grads[::-1][1:], self.grads[::-1], self.weights[::-1]):
            trans_weight = zip(*weight)
            for i in range(len(grad_left)):
                grad_left[i] = sum(
                    self.activate_func_d(lr) * gr * w for lr, gr, w in zip(layer_right, grad_right, next(trans_weight)))
            for i, lr, gr in zip(range(len(grad_right)), layer_right, grad_right):
                weight[i] = [value - self.learning_rate * ll *
                             self.activate_func_d(lr) * gr for value, ll in zip(weight[i], layer_left)]

    def train(self, data: list[tuple[list, list]], *, times: int, log_times: int = 1):
        gap = times if log_times <= 0 else 1 if log_times > times else times // log_times
        for i in range(times):
            for input_data, ideal_output in data:
                self.forward(input_data)
                self.backward(ideal_output)
            print(
                f"loss: {sum(v ** 2 for v in self.grads[-1]) ** 0.5}, weights: {self.weights}") if (i + 1) % gap == 0 else None


def target_func(x):
    return 1 * x**2 + 2 * x + 3


if __name__ == "__main__":
    n = neural([3, 1], learning_rate=0.01,
               activate_func=lambda v: v, activate_func_d=lambda v: 1)
    n.train([([x**2, x, 1], [target_func(x)])
            for x in [-2, -1, 0, 1, 2]], times=100, log_times=5)
