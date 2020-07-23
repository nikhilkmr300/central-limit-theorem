import numpy as np
import distributions

def plot_and_print(dist, num_samples=10000, sample_size=35, plot=True):
    # Plotting population distribution
    if(plot == True):
        dist.plot_population()

    print(f'num_samples={num_samples}, sample_size={sample_size}')

    # Drawing samples from population
    samples = dist.draw_samples(num_samples=num_samples, sample_size=sample_size)

    # Printing details
    print(f'Population mu = {np.round(dist.population_mean(), distributions.ROUND)}')
    print(f'Mean of sample mean E(Xbar) = {np.round(samples.sample_mean_mean(), distributions.ROUND)}')
    print(f'Population standard deviation sigma = {np.round(np.sqrt(dist.population_variance()), distributions.ROUND)}')
    print(f'stddev(Xbar) * sqrt(sample_size) = {np.round(np.sqrt(samples.sample_mean_variance() * samples.sample_size()), distributions.ROUND)}')

    # Plotting sampling distribution of sample means.
    if(plot == True):
        samples.plot_sample_mean_dist(bins=20)

# Setting seed to get reproducible results.
np.random.seed(1)

dists = [
    'distributions.Binomial(n=10, p=0.5)',
    'distributions.Custom(population=np.concatenate([np.array([1]*1000), np.array([2]*500), np.array([3] * 750), np.array([4] * 250), np.array([5] * 300) , np.array([6] * 1000)]))',
    'distributions.Exponential(lam=1)',
    'distributions.Gamma(k=1, theta=2)',
    'distributions.Normal(mu=0, sigma=1)',
    'distributions.Poisson(lam=20)',
    'distributions.Uniform(a=0, b=10)'
]

for dist in dists:
    print(dist[len('distributions.'):] + ':')
    dist = eval(dist)
    plot_and_print(dist, num_samples=10000, sample_size=50, plot=True)
    print('-' * 10)
