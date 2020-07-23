# central-limit-theorem

Scripts to visualize the central limit theorem in action.

The central limit theorem (CLT) is an important theorem in probability and statistics.
While there exist rigorous proofs of CLT, 
I thought it'd be nice to visualize it in action without going into the mathematical nitty-gritties involving characteristic functions.

[Here](https://mathworld.wolfram.com/CentralLimitTheorem.html) is Wolfram's explanation of CLT,
but that can be somewhat esoteric to understand.

I like to explain CLT as follows:

We are given a population with any distribution, with a well-defined mean $\mu$ and variance $\sigma^{2}$.
We then take $m$ samples from the population, with each sample containing $n$ data points each.

The sample mean for a given sample is defined as
$$\overline{X} = \frac{1}{n} \sum_{i=1}^{n} x_{i}$$

There are $m$ samples, and we calculate the sample mean for each of the samples.
So we now have $m$ sample means. We can treat $\overline{X}$ as a random variable itself.
This is called the sampling distribution of the sample means.

The central limit theorem tells us that the sampling distribution of the sample means ($\overline{X}$) is approximately a normal distribution when m is large (m > 35 is usually good enough) with:
* Mean $\mu_{\overline{X}} = \mu$
* Variance $\sigma^{2}_{\overline{X} = \frac{\sigma^{2}}{n}$

where $\mu$ and $\sigma^{2}$ are the population mean and population standard deviation respectively.

Keep in mind that the population may have *any* distribution, so long as that distribution has a well-defined mean $\mu$ and variance $\sigma^{2}$.
So it is pretty remarkable that the sampling distribution of the sample means always obeys a normal distribution for such a distribution.
No wonder the normal distribution is so important!

# Examples

## Exponential distribution

<img src="https://github.com/nikhilkmr300/central-limit-theorem/blob/master/plots/exponential_dist.png" height=320>
<img src="https://github.com/nikhilkmr300/central-limit-theorem/blob/master/plots/exponential_sample_mean_dist.png" height=320>

The exponential distribution with $\lambda=1$ looks pretty different from the normal distribution.

The sampling distribution of the mean looks like a normal distribution. The mean of the sampling distribution of the sample means $\mu_{\overline{X}}=1.003$ is pretty close to the actual population mean $\mu = 1$.
Also, $\sigma^{2}_{\overline{X}} * n = 0.02 * 50 = 1$ is equal to the population variance $\sigma^{2} = 1$.

## Custom distribution

Here is another distribution that doesn't follow any particular formula, and is far from normal, but has a well-defined mean and variance.

<img src="https://github.com/nikhilkmr300/central-limit-theorem/blob/master/plots/custom_dist.png" height=320>
<img src="https://github.com/nikhilkmr300/central-limit-theorem/blob/master/plots/custom_sample_mean_dist.png" height=320>

The sampling distribution looks like a normal distribution, and also has skew and excess kurtosis values close to 0 (as do normal distributions). The mean of the sampling distribution of the sample means $\mu_{\overline{X}}=3.355$ is equal to the actual population mean $\mu = 3.355$.
Also, $\sigma^{2}_{\overline{X}} * n = 0.077 * 50 = 3.850$ is pretty close to the population variance $\sigma^{2} = 3.808$.
