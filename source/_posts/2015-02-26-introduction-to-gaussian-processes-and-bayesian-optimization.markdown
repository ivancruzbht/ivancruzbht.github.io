---
layout: post
title: "Introduction to Gaussian Processes and Bayesian Optimization (Part 1)"
date: 2015-02-26 22:21:22 -0800
comments: true
categories: optimization
---

Gaussian processes have been around for quite some time. They are a set of useful tools widely used in machine learning because they can compute non linear classifiers. Moreover, they are becoming quite important since the advent of Bayesian optimization. And what is Bayesian optimization? In simple terms, it is a methodology to find the best set of <em>hyperparameters</em> in an experiment. 

<!-- more -->

Many machine learning methods have a set of parameters that need to be tuned manually. And I don't mean the inherent parameters of a given model (such as the weight of synapses between neurons in a neural network or the parameters of a kernel), but to those parameters of the prior knowledge. For example, in a neural network, they could be the number of layers, how many neurons each layer have, the learning rate used during training, which activation function should be used, whether to use stochastic gradient decent or AdaGrad as the optimization method, or how many epoch run during training.

It would be great to have an automatic method that finds out the best set of these hyperparameters and not to depend on some machine learning guru (which may not be available when you need it) to set these parameters. 

There are a couple of methods to tune these hyperparameters. The most widely used is grid search. There are also methods based on genetic algorithms and evolutive computing. Nowadays, Bayesian optimization is gaining a lot of attention, and with good reasons: it is elegant, it has solid theoretic foundations, and more importantly, it works quite well.

In this series of entries I will explain the basics of Bayesian optimization. To do that, I'll give an overview of Gaussian processes. And I'll start by giving the basics of normal distributions, covariance matrix and other mathematical background. 

If you really want to learn this stuff with someone that really knows these topics, I highly recommend the Gaussian processes lectures given by Nando de Freitas in his <a href="https://www.youtube.com/channel/UC0z_jCi0XWqI8awUuQRFnyw">youtube channel</a>. For now, I'll keep babbling here for awhile.

<h3>Gaussian Distributions</h3>

A gaussian distribution is a probability distribution where the majority of observations are near of a value, called the mean.

This distributions have a bell-shaped form as shown in the next figure:


<p align="center">{% img center /images/bayesian_optimization/nd_0.png 600 450 'Normal distribution' 'Normal Distribution' %}</p>
~~~ python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

mean = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(-3,3,100)
plt.plot(x,mlab.normpdf(x,mean,sigma))

plt.show()
~~~
<br/>
These type of distribution are described by its <em>mean</em> and its <em>standard deviation</em>. The mean is the average value of an element in the data and it is represented by the greek letter $$\mu$$. The standard deviation can be interpreted as how much a given observation can vary from the mean, and is represented as $$\sigma$$. The <em>variance</em> of a given distribution is the squared standard deviation and it can be used instead. The probability distribution of a univariate gaussian distribution is:

$$\mathcal{N}(x,\mu,\sigma)=\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$


A <em>multivariate gaussian distribution</em> is the same gaussian distribution but more than one random variable. The graphic of a multivariate gaussian distribution with 2 variables would be something like this:

<p align="center">{% img center /images/bayesian_optimization/nd_2.png 600 450 'Multivariate Normal distribution' 'Multivariate Normal Distribution' %}</p>
~~~ python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.gca(projection='3d')
x = np.linspace(-5, 5, 200)
y = x
X,Y = np.meshgrid(x, y)
Z = bivariate_normal(X, Y)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
        linewidth=0, antialiased=False)

ax.set_zlim(0, 0.2)

ax.zaxis.set_major_locator(plt.LinearLocator(10))
ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=7, cmap=plt.cm.coolwarm)

plt.show()
~~~
<br>

The probability distribution of a multivariate gaussian distributions is:

$$\mathcal{N}(\overline{x},\mu,\Sigma)=\frac{1}{\sqrt{(2\pi)^{k}\left | \Sigma \right |}}e^{\frac{1}{2}(\overline{x}-\mu)^{T}\Sigma^{-1}(\overline{x}-\mu)}$$

What is $$\mu$$ and $$\Sigma$$ in this case? I will explain it with an example. Suppose you have a bunch of data represented by two random variables $$X_{1}$$ and $$X_{2}$$ as shown in the next graphic:

<p align="center">{% img center /images/bayesian_optimization/nd_1.png 600 450 'Some random data' 'some random data' %}</p>

~~~ python
import matplotlib.pyplot as plt
import numpy as np

mean = [0,0]
covariance = np.identity(2)
data = np.random.multivariate_normal(mean,covariance,100)
plt.scatter(data[:,0],data[:,1])
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
~~~
<br/>

You want to find out what probability distribution generated that data. Since we have 2 variables involved, the distribution is multivariate. So we have to learn what multivariate distribution fits the data. 

Lets assume that those point were generated under a multivariate gaussian distribution. So each point $$ x \in \mathbb{R}^{2} $$ is a 2-dimentional vector <em>sampled</em> from this distribution. This is expressed like this:

$$ x \sim \mathcal{N}\left ( \mu,\Sigma \right ) $$

So this distribution is our <em>model</em> of the data. Lets see if it is possible infer the distribution. One can observe that the middle of that cluster of points is near to 0 in both variables, i.e. the coordinates $$(0,0)$$, so it is fair to assume that the mean of the distribution is 0 for both random variables. Therefore the mean $$ \mu $$ is a vector where the number of components is the number of random variables:

$$ \mu = \begin{bmatrix} \mu_{1} \\ \mu_{2} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} $$

<h3>Covariance matrix</h3>

What about the variance? Since our distribution is multivariate, our variance is in function of all random variables, and the variance of each random variable no longer depends on itself, but it depends also of the other variables. It is a <em>correlation</em> between these variables. So this correlation, lets call it $$\Sigma$$ is a square matrix of size $$d \times d$$, i.e. $$\Sigma \in \mathbb{R}^{d \times d}$$. In our example we have 2 random variables so $$ \Sigma \in \mathbb{R}^{2 \times 2}$$ and it is the <em>covariance matrix</em>:

$$ \Sigma = \begin{bmatrix} \Sigma_{1,1} && \Sigma_{1,2} \\ \Sigma_{2,1} && \Sigma_{2,2} \end{bmatrix} $$

A particular element $$\Sigma_{m,n}$$ is the covariance between the random variables $$m$$ and $$n$$, i.e. the correlation between $$X_{m}$$ and $$X_{n}$$. If $$m=n$$ then $$\Sigma_{m,n}$$ is the variance of $$X_{m}$$.

How should one interpret this correlation? Lets look at the next graphic:

<p align="center">{% img center /images/bayesian_optimization/nd_3.png 600 450 'Some random data' 'some random data' %}</p>
<br/>

The red dotted line is $$X_{1}=1.5$$. At this point, $$X_{2}$$ can be both positive or negative, so $$X_{1}$$ does not tell us too much about $$X_{2}$$, or in other words, $$X_{1}$$ is not correlated to $$X_{2}$$.

On the other hands, lets look the next figure:

<p align="center">{% img center /images/bayesian_optimization/nd_4.png 600 450%}</p>

~~~ python
import matplotlib.pyplot as plt
import numpy as np

mean = [0,0]
covariance = np.identity(2)
covariance[0,1] = 0.9
covariance[1,0] = 0.9
data = np.random.multivariate_normal(mean,covariance,100)
plt.scatter(data[:,0],data[:,1])
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.xlabel("X1")
plt.ylabel("X2")
plt.axhline(y=0, color='k',linestyle='--')
plt.axvline(x=0, color='k',linestyle='--')
plt.axvline(x=1.5, color='r',linestyle=':')
plt.show()
~~~
<br/>

At $$X_{1}=1.5$$, $$X_{2}>0$$ for every point. Moreover one can observe that as the value of $$X_{2}$$ increases for each point, so does the value of $$X_{2}$$. Likewise, if $$X_{1}$$ decreases, so does $$X_{2}$$. We can conclude that there is a linear correlation between both variables. So for the case the variables are not correlated, $$\Sigma$$ could be:

$$ \Sigma = \begin{bmatrix} 1 && 0 \\ 0 && 1 \end{bmatrix} $$

where $$\Sigma_{1,2}=\Sigma_{2,1}=0$$, i.e. there is no correlation between $$X_{1}$$ and $$X_{2}$$. For the case where there is a correlation, $$\Sigma$$ could be something like this:

$$ \Sigma = \begin{bmatrix} 1 && 0.9 \\ 0.9 && 1 \end{bmatrix} $$

If you check the python code of the previous figure, this is the $$\Sigma$$ used.

<h3>Maximum Likelihood Estimation</h3>

By now you must be asking yourself "All this is nice, but how can we compute $$\mu$$ and $$\Sigma$$ given a bunch of data?"

Lets suppose that we know the data model is a probability distribution, (in our case, a Gaussian probability distribution). So it is fair to assume that these points were the most likely to be sampled because they have the highest probability. In other words, a model (parametrized by $$\mu$$ and $$\Sigma$$) that explains the data better is the model chosen. This principle is called Maximum Likelihood Estimation and it can be exploited to infer $$\mu$$ and $$\Sigma$$.

How to get $$\mu$$? Lets look at the next figure:

<p align="center">{% img center /images/bayesian_optimization/nd_5.png 600 450%}</p>

The value of $$\mu$$ is where the probability is the highest. The red line represents the tangent at that point and its slope is 0. Therefore the value of $$\mu$$ is where the derivative of the probability function is 0. Converting the Gaussian probability distribution function to logarithmic space to facilitate the math, doing a bunch of derivations and setting the derivate to zero, $$\mu$$ is:

$$\mu_{ML}=\frac{1}{N}\sum_{n=1}^{N}x_{n}$$

which is the average of the data. Doing a similar process, $$\Sigma$$ is:

$$\Sigma_{ML} = \frac{1}{N}\sum_{n=1}^{N}(x_{n}-\mu)(x_{n}-\mu)^{T}$$

The $$_{ML}$$ subindex just indicates that these parameters work under the principle of maximum likelihood expectation.

<h3>What's next?</h3>

In the next entry I will explain joint Gaussian distributions and conditional Gaussian distributions.
