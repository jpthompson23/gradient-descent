from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * np.power(x - mu, 2.0) / (np.power(sigma, 2.0)))


class GaussianModel(object):
    def __init__(self, rate=0.01):
        self.mu = 0.0
        self.sigma = 1.0
        self.rate = rate

    def learn_params(self, x, y):
        test_mu = [self.mu - self.rate, self.mu + self.rate]
        test_sigma = [self.sigma - self.rate, self.sigma + self.rate]
        test_params = [
            (test_mu[0], test_sigma[0]),
            (test_mu[0], test_sigma[1]),
            (test_mu[1], test_sigma[0]),
            (test_mu[1], test_sigma[1])
        ]

        errors = [
            np.sum((y - gaussian(x, *p)) ** 2) for p in test_params
        ]

        min_pos = errors.index(min(errors))

        self.mu, self.sigma = test_params[min_pos]

        print(self.mu, self.sigma)

    def fit(self, x, y):
        epochs = 5000
        batch_size = 500
        for i in (trng := trange(epochs)):
            rand_samp = np.random.randint(0, x.shape[0], size=batch_size)
            x_samp = x[rand_samp]
            y_samp = y[rand_samp]

            self.learn_params(x_samp, y_samp)

            y_pred = self.forward(x)
            error = np.sum((y - y_pred)**2)
            trng.set_description(f"error: {error:.2f} / progress")

    def forward(self, x):
        return gaussian(x, self.mu, self.sigma)


def main():
    x = np.arange(0, 10, 0.01)
    noise = np.random.uniform(low=-0.2, high=0.3, size=x.shape)
    y = gaussian(x, 5.0, 2.0) + noise
    y = np.asarray([0 if value < 0 else value for value in y])
    plt.plot(x, y, "ro", mfc='none')

    model = GaussianModel()
    model.fit(x, y)
    print(model.mu, model.sigma)
    y_pred = model.forward(x)
    plt.plot(x, y_pred)
    plt.show()


if __name__ == "__main__":
    main()
