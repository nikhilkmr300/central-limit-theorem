import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# Rounding to ROUND number of decimal places
ROUND = 3

class Distribution:
    """ Class to represent a generalized discrete/continuous distribution. """

    def reset_params(self):
        """ Resets parameters of the distribution. """
        pass

    def draw_samples(self, num_samples, sample_size):
        """
        Draws num_samples number of samples, each of size sample_size
        from the population.
        Returns a 2D NumPy array, whose each row corresponds to a sample
        containing sample_size number of data points.
        """
        pass

    def population_mean(self):
        """ Returns mean of the population. """
        pass

    def population_median(self):
        """ Returns median of the population. """
        pass

    def population_mode(self):
        """ Returns mode of the population. """
        pass

    def population_variance(self):
        """ Returns population variance (unbiased). """
        pass

    def population_skew(self):
        """ Returns skewness of the population. """
        pass

    def population_kurtosis(self):
        """
        Returns Fisher kurtosis (excess kurtosis) of the population.
        Fisher kurtosis = Pearson kurtosis - 3
        """
        pass

    def plot_population(self):
        """ Plots the population PDF / PMF / frequency distribution. """
        pass

class Samples:
    """ Class to represent samples drawn from a population. """

    def __init__(self, samples):
        """
        samples is a 2D array of shape (num_samples, sample_size). Each row in
        samples corresponds to one sample of size sample_size.
        """
        self.samples = np.array(samples)

    def __str__(self):
        return f'Samples({self.samples})'

    def num_samples(self):
        """ Returns number of samples. """
        return self.samples.shape[0]

    def sample_size(self):
        """ Returns number of data points in each sample. """
        return self.samples.shape[1]

    def sample_mean(self):
        """ Returns an array of sample means, taken for each sample. """
        return np.mean(self.samples, axis=1)

    def sample_median(self):
        """ Returns an array of sample median, taken for each sample. """
        return np.median(self.samples, axis=1)

    def sample_mode(self):
        """ Returns an array of sample modes, taken for each sample. """
        return getattr(scipy.stats.mode(self.samples, axis=1), 'mode').flatten()

    def sample_variance(self, ddof=1):
        """
        Returns an array of sample variances, taken for each sample. ddof is
        taken as 1 by default as calculating sample variance.
        """
        return np.var(self.samples, axis=1, ddof=ddof)

    def sample_skew(self, bias=True):
        """ Returns an array of sample skewness values, taken for each sample. """
        return scipy.stats.skew(self.samples, axis=1, bias=bias)

    def sample_kurtosis(self, fisher=True, bias=True):
        """ Returns an array of sample kurtosis values, taken for each sample. """
        return scipy.stats.kurtosis(self.samples, axis=1, fisher=fisher, bias=bias)

    def describe(self, ddof=1, bias=True):
        """ Returns a summary of statistics, taken for each sample. """
        return scipy.stats.describe(self.samples, axis=1, ddof=ddof, bias=bias)

    def plot_samples(self, bins=None, range=None, align='mid', histtype='step', xlabel='x', ylabel='Frequency', title=None, legend=True):
        """ Plots a histogram for each sample. """
        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        for i, sample in enumerate(self.samples):
            plt.hist(sample, bins=bins, range=range, align=align, histtype=histtype, label=f'Sample {i + 1}')
        if(legend == True):
            plt.legend()
        plt.show()

    def sample_mean_mean(self):
        """ Returns mean of the sampling distribution of the sample mean. """
        return np.mean(np.mean(self.samples, axis=1))

    def sample_mean_median(self):
        """ Returns median of the sampling distribution of the sample mean. """
        return np.median(np.mean(self.samples, axis=1))

    def sample_mean_mode(self):
        """ Returns mode of the sampling distribution of the sample mean. """
        return getattr(scipy.stats.mode(np.mean(self.samples, axis=1)), 'mode').flatten()

    def sample_mean_variance(self, ddof=1):
        """ Returns variance of the sampling distribution of the sample mean. """
        return np.var(np.mean(self.samples, axis=1), ddof=ddof)

    def sample_mean_skew(self, bias=True):
        """ Returns skewness of the sampling distribution of the sample mean. """
        return scipy.stats.skew(np.mean(self.samples, axis=1), bias=bias)

    def sample_mean_kurtosis(self, fisher=True, bias=True):
        """ Returns kurtosis of the sampling distribution of the sample mean. """
        # Use fisher=True to calculate excess kurtosis and fisher=False to
        # calculate Pearson kurtosis.
        return scipy.stats.kurtosis(np.mean(self.samples, axis=1), fisher=fisher, bias=bias)

    def plot_sample_mean_dist(self, bins=None, range=None, align='mid', histtype='bar', color='b', xlabel='x', ylabel='Frequency'):
        """ Plots sampling distribution of the sample mean. """

        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'Sampling distribution of sample means' + ' $\overline{X}$\n' +
        '\n$\mu(\overline{X})$' + f'={np.round(self.sample_mean_mean(), ROUND)} | ' +
        'median($\overline{X}$)' + f'={np.round(self.sample_mean_median(), ROUND)} | ' +
        'mode($\overline{X}$)' + f'={np.round(self.sample_mean_mode(), ROUND)} | ' +
        '$\sigma^{2}_{\overline{X}}$' + f'={np.round(self.sample_mean_variance(), ROUND)}\n' +
        'skew($\overline{X}$)' + f'={np.round(self.sample_mean_skew(), ROUND)} | ' +
        'kurtosis($\overline{X}$)' + f'={np.round(self.sample_mean_kurtosis(), ROUND)}' +
        f'\nnum_samples={self.num_samples()} | sample_size={self.sample_size()}')
        plt.hist(self.sample_mean(), bins=bins, range=range, align=align, histtype=histtype, color=color)
        plt.show()

class Binomial(Distribution):
    """ Class to represent a binomial distribution. """

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def reset_params(self, n, p):
        self.n = n
        self.p = p

    def draw_samples(self, num_samples=1, sample_size=35):
        return Samples(np.random.binomial(n=self.n, p=self.p, size=(num_samples, sample_size)))

    def population_mean(self):
        return self.n * self.p

    def population_median(self):
        return (np.floor(self.n * self.p) + np.ceil(self.n * self.p)) / 2

    def population_mode(self):
        return np.floor((self.n + 1) * self.p)

    def population_variance(self):
        return self.n * self.p * (1 - self.p)

    def population_skew(self):
        return (1 - 2 * self.p) / np.sqrt(self.population_variance())

    def population_kurtosis(self):
        return (1 - 6 * self.p * (1 - self.p)) / self.population_variance()

    def plot_population(self, step=1, xlabel='x', ylabel='PMF', color='b'):
        x = np.arange(0, self.n + 0.0001, step=step)
        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'Binomial\n$\mu$={np.round(self.population_mean(), ROUND)} | median={np.round(self.population_median(), ROUND)} | mode={np.round(self.population_mode(), ROUND)} | $\sigma^{2}$={np.round(self.population_variance(), ROUND)}\nskew={np.round(self.population_skew(), ROUND)} | kurtosis={np.round(self.population_kurtosis(), ROUND)}')
        plt.plot(x, scipy.stats.binom.pmf(x, self.n, self.p), color=color)
        plt.show()

class Custom(Distribution):
    """ Class to represent a custom/user-defined distribution. """

    def __init__(self, population):
        """
        Initializes custom/user-defined distribution (population).

        Parameter:
        population (arraylike): Array of values in the population.
        """
        self.population = np.array(population)

    def reset_params(self, population):
        """
        Resets population with new values.

        Parameter:
        Same as for __init__.
        """
        self.population = np.array(population)

    def draw_samples(self, num_samples=1, sample_size=35):
        choices = np.zeros((num_samples, sample_size))
        for i in range(num_samples):
            choices[i] = np.random.choice(self.population, size=sample_size)

        return Samples(choices)

    def population_mean(self):
        return np.mean(self.population)

    def population_median(self):
        return np.median(self.population)

    def population_mode(self):
        return getattr(scipy.stats.mode(self.population), 'mode').flatten()

    def population_variance(self):
        return np.var(self.population, ddof=0)  # ddof = 0 as calculating for population, not sample

    def population_skew(self, bias=True):
        return scipy.stats.skew(self.population, bias=bias)

    def population_kurtosis(self, bias=True):
        # To maintain consistency across distributions, calculating Fisher kurtosis for population.
        return scipy.stats.kurtosis(self.population, fisher=True, bias=bias)

    def plot_population(self, bins=None, range=None, align='mid', histtype='bar', color='b', xlabel='x', ylabel='Frequency'):
        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'Custom\n$\mu$={np.round(self.population_mean(), ROUND)} | median={np.round(self.population_median(), ROUND)} | mode={np.round(self.population_mode(), ROUND)} | $\sigma^{2}$={np.round(self.population_variance(), ROUND)}\nskew={np.round(self.population_skew(), ROUND)} | kurtosis={np.round(self.population_kurtosis(), ROUND)}')
        plt.hist(self.population, bins=bins, range=range, align=align, histtype=histtype, color=color)
        plt.show()

class Exponential(Distribution):
    def __init__(self, lam):
        self.lam = lam

    def reset_params(self, lam):
        self.lam = lam

    def draw_samples(self, num_samples=1, sample_size=35):
        return Samples(np.random.exponential(scale=(1 / self.lam), size=(num_samples, sample_size)))

    def population_mean(self):
        return 1 / self.lam

    def population_median(self):
        return np.log(2) / self.lam

    def population_mode(self):
        return 0

    def population_variance(self):
        return 1 / (self.lam)**2

    def population_skew(self):
        return 2

    def population_kurtosis(self):
        return 6

    def plot_population(self, xlim_upper=20, num_points=100, xlabel='x', ylabel='PDF', color='b'):
        x = np.linspace(0, xlim_upper, num_points)
        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim([0, xlim_upper])
        plt.title(f'Exponential\n$\mu$={np.round(self.population_mean(), ROUND)} | median={np.round(self.population_median(), ROUND)} | mode={np.round(self.population_mode(), ROUND)} | $\sigma^{2}$={np.round(self.population_variance(), ROUND)}\nskew={np.round(self.population_skew(), ROUND)} | kurtosis={np.round(self.population_kurtosis(), ROUND)}')
        plt.plot(x, scipy.stats.expon.pdf(x, scale=(1 / self.lam)), color=color)
        plt.show()

class Gamma(Distribution):
    def __init__(self, k, theta):
        self.k = k
        self.theta = theta

    def reset_params(self, k, theta):
        self.k = k
        self.theta = theta

    def draw_samples(self, num_samples=1, sample_size=35):
        return Samples(np.random.gamma(self.k, scale=self.theta, size=(num_samples, sample_size)))

    def population_mean(self):
        return self.k * self.theta

    def population_median(self):
        return np.nan

    def population_mode(self):
        if(self.k >= 1):
            return (self.k - 1) * self.theta
        return 0

    def population_variance(self):
        return self.k * self.theta ** 2

    def population_skew(self):
        return 2 / np.sqrt(self.k)

    def population_kurtosis(self):
        return 6 / self.k

    def plot_population(self, xlim_upper=20, num_points=100, xlabel='x', ylabel='PDF', color='b'):
        x = np.linspace(0, xlim_upper, num_points)
        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim([0, xlim_upper])
        plt.title(f'Gamma\n$\mu$={np.round(self.population_mean(), ROUND)} | median={np.round(self.population_median(), ROUND)} | mode={np.round(self.population_mode(), ROUND)} | $\sigma^{2}$={np.round(self.population_variance(), ROUND)}\nskew={np.round(self.population_skew(), ROUND)} | kurtosis={np.round(self.population_kurtosis(), ROUND)}')
        plt.plot(x, scipy.stats.gamma.pdf(x, self.k, scale=self.theta), color=color)
        plt.show()

class Normal(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def reset_params(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def draw_samples(self, num_samples=1, sample_size=35):
        return Samples(np.random.normal(loc=self.mu, scale=self.sigma, size=(num_samples, sample_size)))

    def population_mean(self):
        return self.mu

    def population_median(self):
        return self.mu

    def population_mode(self):
        return self.mu

    def population_variance(self):
        return self.sigma ** 2

    def population_skew(self):
        return 0

    def population_kurtosis(self):
        return 0

    def plot_population(self, lower_sigma=-3, upper_sigma=3, num_points=100, xlabel='x', ylabel='PDF', color='b'):
        x = np.linspace(self.mu + lower_sigma * self.sigma, self.mu + upper_sigma * self.sigma + 0.0001, num_points)
        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'Normal\n$\mu$={np.round(self.population_mean(), ROUND)} | median={np.round(self.population_median(), ROUND)} | mode={np.round(self.population_mode(), ROUND)} | $\sigma^{2}$={np.round(self.population_variance(), ROUND)}\nskew={np.round(self.population_skew(), ROUND)} | kurtosis={np.round(self.population_kurtosis(), ROUND)}')
        plt.plot(x, scipy.stats.norm.pdf(x, self.mu, self.sigma), color=color)
        plt.show()

class Poisson(Distribution):
    def __init__(self, lam):
        self.lam = lam

    def reset_params(self, lam):
        self.lam = lam

    def draw_samples(self, num_samples=1, sample_size=35):
        return Samples(np.random.poisson(lam=self.lam, size=(num_samples, sample_size)))

    def population_mean(self):
        return self.lam

    def population_median(self):
        return np.floor(self.lam + (1 / 3) - (0.02 / self.lam))

    def population_mode(self):
        return np.floor(self.lam)

    def population_variance(self):
        return self.lam

    def population_skew(self):
        return 1 / np.sqrt(self.lam)

    def population_kurtosis(self):
        return 1 / self.lam

    def plot_population(self, step=1, xlabel='x', ylabel='PMF', color='b'):
        x = np.arange(0, 2 * self.lam + 0.001, step=step)
        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'Poisson\n$\mu$={np.round(self.population_mean(), ROUND)} | median={np.round(self.population_median(), ROUND)} | mode={np.round(self.population_mode(), ROUND)} | $\sigma^{2}$={np.round(self.population_variance(), ROUND)}\nskew={np.round(self.population_skew(), ROUND)} | kurtosis={np.round(self.population_kurtosis(), ROUND)}')
        plt.plot(x, scipy.stats.poisson.pmf(x, self.lam), color=color)
        plt.show()

class Uniform(Distribution):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def reset_params(self, a, b):
        self.a = a
        self.b = b

    def draw_samples(self, num_samples=1, sample_size=35):
        return Samples(np.random.uniform(low=self.a, high=self.b, size=(num_samples, sample_size)))

    def population_mean(self):
        return (self.a + self.b) / 2

    def population_median(self):
        return (self.a + self.b) / 2

    def population_mode(self):
        return (self.a + self.b) / 2

    def population_variance(self):
        return (self.b - self.a)**2 / 12

    def population_skew(self):
        return 0

    def population_kurtosis(self):
        return (- 6 / 5)

    def plot_population(self, step=1, num_points=100, xlabel='x', ylabel='PDF', color='b'):
        x = np.linspace(self.a, self.b, num_points)
        plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'Uniform\n$\mu$={np.round(self.population_mean(), ROUND)} | median={np.round(self.population_median(), ROUND)} | mode={np.round(self.population_mode(), ROUND)} | $\sigma^{2}$={np.round(self.population_variance(), ROUND)}\nskew={np.round(self.population_skew(), ROUND)} | kurtosis={np.round(self.population_kurtosis(), ROUND)}')
        plt.plot(x, scipy.stats.uniform.pdf(x, loc=self.a, scale=self.b - self.a), color=color)
        plt.show()
