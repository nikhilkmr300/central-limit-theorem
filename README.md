# central-limit-theorem

Scripts to visualize the central limit theorem in action.

The central limit theorem (CLT) is an important theorem in probability and statistics.
While there exist rigorous proofs of CLT, 
I thought it'd be nice to visualize it in action without going into the mathematical nitty-gritties involving characteristic functions.

[Here](https://mathworld.wolfram.com/CentralLimitTheorem.html) is Wolfram's explanation of CLT,
but that can be somewhat esoteric to understand.

I like to explain CLT as follows:

We are given a population with any distribution, with a well-defined mean <img src="/tex/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode&sanitize=true" align=middle width=9.90492359999999pt height=14.15524440000002pt/> and variance <img src="/tex/c1e6c992a6df5a9dd8119ccad0590805.svg?invert_in_darkmode&sanitize=true" align=middle width=16.535428799999988pt height=26.76175259999998pt/>.
We then take <img src="/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/> samples from the population, with each sample containing <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> data points each.

The sample mean for a given sample is defined as
<p align="center"><img src="/tex/ff8e781ce80177f17f1ccc8fcb2a5808.svg?invert_in_darkmode&sanitize=true" align=middle width=93.9078954pt height=44.89738935pt/></p>

There are <img src="/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/> samples, and we calculate the sample mean for each of the samples.
So we now have <img src="/tex/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.433101099999991pt height=14.15524440000002pt/> sample means. We can treat <img src="/tex/ba22e75e3fae4f305e38dc9341ed5e5f.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=27.725679300000007pt/> as a random variable itself.
This is called the sampling distribution of the sample means.

The central limit theorem tells us that the sampling distribution of the sample means (<img src="/tex/ba22e75e3fae4f305e38dc9341ed5e5f.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=27.725679300000007pt/>) is approximately a normal distribution when m is large (m > 35 is usually good enough) with:
* Mean <img src="/tex/fa4bc36a88895a093c2a350049f7acf3.svg?invert_in_darkmode&sanitize=true" align=middle width=54.224101799999985pt height=14.15524440000002pt/>
* Variance <img src="/tex/5eed8c9a57338c194a2a89660ea848be.svg?invert_in_darkmode&sanitize=true" align=middle width=46.54230074999999pt height=26.76175259999998pt/>

where <img src="/tex/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode&sanitize=true" align=middle width=9.90492359999999pt height=14.15524440000002pt/> and <img src="/tex/c1e6c992a6df5a9dd8119ccad0590805.svg?invert_in_darkmode&sanitize=true" align=middle width=16.535428799999988pt height=26.76175259999998pt/> are the population mean and population standard deviation respectively.

Keep in mind that the population may have *any* distribution, so long as that distribution has a well-defined mean <img src="/tex/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode&sanitize=true" align=middle width=9.90492359999999pt height=14.15524440000002pt/> and variance <img src="/tex/c1e6c992a6df5a9dd8119ccad0590805.svg?invert_in_darkmode&sanitize=true" align=middle width=16.535428799999988pt height=26.76175259999998pt/>.
So it is pretty remarkable that the sampling distribution of the sample means always obeys a normal distribution for such a distribution.
No wonder the normal distribution is so important!

# Examples

## Exponential distribution

<img src="https://github.com/nikhilkmr300/central-limit-theorem/blob/master/plots/exponential_dist.png" height=320>
<img src="https://github.com/nikhilkmr300/central-limit-theorem/blob/master/plots/exponential_sample_mean_dist.png" height=320>

The exponential distribution with <img src="/tex/013c1ac9d799485f5896316f664b6365.svg?invert_in_darkmode&sanitize=true" align=middle width=39.72592304999999pt height=22.831056599999986pt/> looks pretty different from the normal distribution.

The sampling distribution of the mean looks like a normal distribution. The mean of the sampling distribution of the sample means <img src="/tex/feeca6f87a5b8ece2b5f5862ed5ed1fe.svg?invert_in_darkmode&sanitize=true" align=middle width=81.76223879999999pt height=21.18721440000001pt/> is pretty close to the actual population mean <img src="/tex/aeadc1ef407a8f16104b5f76e0114552.svg?invert_in_darkmode&sanitize=true" align=middle width=40.04176439999999pt height=21.18721440000001pt/>.
Also, <img src="/tex/ab6de4e7bf31998aa9640826579accfe.svg?invert_in_darkmode&sanitize=true" align=middle width=160.52328104999998pt height=26.76175259999998pt/> is equal to the population variance <img src="/tex/b9666e2986b2e31189cafd4eea2a50e2.svg?invert_in_darkmode&sanitize=true" align=middle width=47.49418244999999pt height=26.76175259999998pt/>.

## Custom distribution

Here is another distribution that doesn't follow any particular formula, and is far from normal, but has a well-defined mean and variance.

<img src="https://github.com/nikhilkmr300/central-limit-theorem/blob/master/plots/custom_dist.png" height=320>
<img src="https://github.com/nikhilkmr300/central-limit-theorem/blob/master/plots/custom_sample_mean_dist.png" height=320>

The sampling distribution looks like a normal distribution, and also has skew and excess kurtosis values close to 0 (as do normal distributions). The mean of the sampling distribution of the sample means <img src="/tex/5df527ca07b9c29541477f8e9693f8b8.svg?invert_in_darkmode&sanitize=true" align=middle width=81.76223879999999pt height=21.18721440000001pt/> is equal to the actual population mean <img src="/tex/10e4ec9a66f83db6cba768aa244ce7e7.svg?invert_in_darkmode&sanitize=true" align=middle width=69.26561729999999pt height=21.18721440000001pt/>.
Also, <img src="/tex/56c208ac52c2dc44e8ac14446eb6c279.svg?invert_in_darkmode&sanitize=true" align=middle width=197.96634329999998pt height=26.76175259999998pt/> is pretty close to the population variance <img src="/tex/4bb7e1971d1da2f7a76347170d4f25b8.svg?invert_in_darkmode&sanitize=true" align=middle width=76.71803369999999pt height=26.76175259999998pt/>.
