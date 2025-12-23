---
layout: post
title: "Stochastic Calculus Basics for Traders"
date: 2025-12-07 12:00:00
categories: [quantitative-finance, stochastic-calculus]
tags: [ito, brownian-motion, gbm, derivatives]
giscus_comments: true
related_posts: false
related_publications: true
---

In order to mathematically model stock price, we start by thinking about it as a random quantity that varies over time.  Such evolving random phenomena can be modelled as a stochastic (or random) process, which is essentially a sequence of random variables indexed by time. 

A random variable (RV) takes value from a set of outcomes (sample space) as a result of some random event, for instance roll of a dice or toss of a coin (in case of stock prices this random event is multiple participants quoting and trading in the market). A stochastic process is a sequence of random variables indexed by numbers from an index set. The index set can be discrete or continuous. For modelling stock price $S(t)$ index set is continuous time $t$.

Once we have decided to model price as stochastic process, next question is what kind of distribution does each RV in this sequence follow. We will start with a simple random process and gradually build upon it. We assume that change in stock price in a given interval is normally distributed with mean 0 and variance $\Delta t$. Also, we assume that price change in two non-overlapping intervals is independent of each other. These are simplifying assumptions, prices in real life exhibit momentum and drift (challenging the independence and $mean = 0$ assumption respectively) among other complexities. As we build the theory, we will see in later notes some stochastic processes that can take care of these assumptions.

### Brownian Motion
Given the setup above, we now describe a simple stochastic process for modelling stock prices. We assume that the process starts at $0$, and at each moment the change in its value is normally distributed and independent of previous changes. This is a known and well studied random process called Brownian motion. We can define it as:

1. $B(0)=0$
2. $B(t)-B(s) \sim N(0,t-s)$
3. $B(t_i) - B(s_i)$ are independent over non overlapping intervals

<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-2 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/brownian_motion_100.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

As we are interested in change of $B(t)$ in a interval of time $\Delta t$, we need to understand some of the properties of the function $B(t)$. Two basic questions that come to mind are:
1. Is this function continuous? 
2. Is this function smooth/differentiable?

##### 1. Continuity
For a non-stochastic function, we can think of continuity as no jumps in the function value i.e. a function $f(t)$ is continuous at $t_0$ if its value doesn't change if we approach it from left or right hand side i.e. 

$$
\begin{align}
\lim_{h\to 0} f(t_o+h) - f(t_o) = 0
\end{align}
$$

For a stochastic function, this statement would be made in probability i.e. the probability of a jump in value of $B(t)$ in small time interval $h$ tends to $0$ as $h \rightarrow 0$ i.e. for any fixed $\epsilon > 0$

$$
\begin{align}
lt_{h \rightarrow 0} P(|B(t+h) - B(t)| > \epsilon) &= 2(1 - \phi(\frac{\epsilon}{\sqrt{h}})) \rightarrow 0
\end{align}
$$
where $\phi(x)$ is the cdf of $N(0,1)$ standard normal distribution. 

##### 2. Smoothness
For a non-stochastic function, we define smoothness (differentiability) at a point $x$ as ability draw a slope i.e. below limit should exist:

$$
\begin{align}
\lim_{h\to 0}\frac{f(x + h) - f(x)}{h} 
\end{align}
$$

What about brownian motion, is $B(t)$ is differentiable? For stochasstic function similar to continuity, we check limit in probability. Note, $\frac{B(t+h)-B(t)}{h} \sim N(0, \frac{1}{h})$. For small $h$ this distribution has large variance, hence the slope is not well defined. We can also see this as for any fixed $M$, below limit doesn't converge:

$$
\begin{align}
\lim_{h\to 0} P\left(\frac{|B(t+h)-B(t)|}{h} > M\right) &=  2(1 - \phi(\epsilon\sqrt{h})) \rightarrow 1
\end{align}
$$

This is not intuitive. One would suppose the niceties of the normal distribution to play out and not let $B$ jump around a lot in small duration. But as the limit shows the brownian path is indeed extremely rough. As $h\rightarrow 0$,  change in $B$ could be arbitrarily large and that means we can't draw a line along $B(t)$ and define a slope. So, brownian motion is continuous i.e. no sudden jumps but it just doesn't follow a straight line path no matter how deep you zoom. It keeps on changing the direction. Think of it as infinitely wiggly line.

<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-2 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/brownian_motion.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Infact, one may feel such a line should have infinite length. Let's say we sum $\| \Delta B(t)\|$ over time interval $[0,T]$ by breaking it into $N$ small steps of length $h = \frac{T}{N}$ in time i.e. we calculate: $\sum \| \Delta B(t)\|$

Length of each $\|\Delta B(t)\| = \sqrt{h^2 + (B(t+h)-B(t))^2}$. As $B(t+h)-B(t) \sim N(0,h) = \sqrt{h} N(0,1)$ it implies $\|B(t+h)-B(t)\| \approx O(\sqrt{h})$ (we can check this by computing $P(\|B(t+h)-B(t)\| < h)$). Note, $\sqrt{h} \ggg h$ for small values of $h$ (because $\lim_{h\to0} \frac{\sqrt{h}}{h}\to \infty $). This means, $(B(t+h)-B(t))^2 = O(h)$ term dominates the $\|\Delta B(t)\|$ computataion for small $h$.   

$$
\begin{align}
\label{db2}
|\Delta B(t)| = \sqrt{h^2 + (B(t+h)-B(t))^2} \approx |B(t+h)-B(t)| \approx \sqrt{h}
\end{align}
$$

This takes the sum to infinity as $N$ steps grow:

$$
\begin{align*}
\sum_{i} |\Delta B(t)| &= \sum_i \sqrt{h} = N\sqrt{h} \\
&= N\sqrt{\frac{T}{N}}\\
&= \sqrt{NT} \frac{\to}{N\to \infty} \infty 
\end{align*}
$$

This sum is called total varitaion of a function and we see that brownian motion has infinite total variation. On the other hand, for well behaved smooth continuous non stochastic functions total variation is finite. 

###### Quadratic Variation
From total variation computation we can note that if instead of $\sum \|\Delta B(t)\|$ we do $\sum (\Delta B(t))^2$, we will get a finite quantity. 

$$
\begin{align*}
\sum_{i} (\Delta B(t))^2 &= \sum_i h = Nh \\
&= T
\end{align*}
$$

This is called the quadratic variation of a function and for brownian motion it is finite and equal to length of time interval. Btw, note all these are high probability results, since function itself is stochastic but in a handwavy way we use the $=$ sign. Lets see quadratic variation for a non-stochastic function. Consider function $f(x) = x$ and interval $t=0$ to $t=T$. 

$$
\begin{align}
QV=\sum \{ f(t_{i+1}) - f(t_i) \}^2 = \sum (T/N)^2 = T^2/N
\end{align}
$$

As $N\rightarrow\infty$, $QV\rightarrow0$. This is understandable, as we refine our interval the quadratic terms will be too small and thus the sum is 0. This holds true for all well behaved functions that we generally encounter. We can also write using equation \ref{db2} $\Delta B^2 = \Delta t$.

### Function of stochastic process
Securities payoffs can often be expressed as a function of stock price. While stock price is modelled as a stochastic process, the function itself is well behaved. For instance, payoff of a call option with strike $K$ at expiry is: 

$$
\begin{align}
f(S_T) &= max(0, S_T - K)
\end{align}
$$

This motivates studying functions of stochastic process and how to compute small changes in function value. Let's start with $f(x)$ where $x$ is non-stochastic. 
We can use Taylor's expansion to compute change in function value over small change in $x$:

$$
\begin{align*}
    \Delta f(x) &= f(x+\Delta x) - f(x) \\
    &= f(x) + f'(x) \Delta x + \frac{1}{2} f''(x) \Delta x^2 - f(x)
\end{align*}
$$

Ignoring quadratic and above terms for small $\Delta x$, we get:
$$
\begin{align*}
    \Delta f(x) &= f'(x) \Delta x 
\end{align*}
$$

We can use similar approach to compute $\Delta f(B_t)$, but we can't ignore the quadratic term since we know from equation \ref{db2} that $\Delta B^2 = \Delta t$. 

$$
\begin{align*}
    \Delta f(B_t) &= f(B_t+\Delta B_t) - f(B_t) \\
    &= f(B_t) + f'(B_t) \Delta B_t + \frac{1}{2} f''(B_t) \Delta {B_t}^2 - f(B_t) \\
    &= f'(B_t) \Delta B_t + \frac{1}{2} f''(B_t) \Delta t
\end{align*}
$$

We can now define $\Delta B_t = dB$ for small $dt$ and drop the $t$ notation, this leads us to one of our main result (ito1):
$$
\begin{align}
\label{eq:ito1}
    df= f'(B)dB + \frac{1}{2} f''(B) dt
\end{align}
$$

##### Ito's Lemma: Extension to $f(t,X)$
So far we saw the case where $f$ is a single parameter function. Through dependence on $B$, $f$ was indirectly dependent on time through stochasticity of $B$. However, in some other commonly encountered scenarios $f$ may have an explicit dependence on time as well. For instance, for call option at strike $K$ with time to expiry $t$, its price can be expressed as:
$$
\begin{align*}
C(S_T, t) = e^{-r(T-t)}E[max(S_T-K, 0)]
\end{align*}
$$

In this case we get additional term for $\frac{\partial f}{\partial t}$

$$
\begin{align}
\label{eq:ito2}
    df(t,X)= \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial X}dX + \frac{1}{2} \frac{\partial^2 f}{\partial X^2} dX^2 
\end{align}
$$

For a process defined simply as $X = B_t$ or $dX = dB$ we can replace $dX^2 = dt$

$$
\begin{align*}
\label{eq:ito3}
    df(t,X)= \left(\frac{\partial f}{\partial t}+\frac{1}{2} \frac{\partial^2 f}{\partial X^2}\right) dt + \frac{\partial f}{\partial X}dB
\end{align*}
$$

For another slightly involved stochastic process defined as $dX_t = \mu_t dt + \sigma_t dB$ we can substitute in equation \ref{eq:ito2} to get:
$$
\begin{align}
    df(t,X)&= \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial X}\left( \mu_t dt + \sigma_t dB\right) + \frac{1}{2} \frac{\partial^2 f}{\partial X^2} \left(\mu_t dt + \sigma_t dB\right)^2 \\
    &=\left(\frac{\partial f}{\partial t} + \mu_t \frac{\partial f}{\partial X} + \frac{1}{2} {\sigma}^2_t \frac{\partial^2 f}{\partial X^2}\right)dt + \sigma_t \frac{\partial f}{\partial X}dB
\end{align}
$$
Here we have ignored higher order $dt^2$ and $dBdt$ terms.


##### Extension to Geometric Brownian Motion (GBM)
Further, we extend application of Ito's calculus to GBM which is often used to model stock prices. Under GBM, instantaneous stock returns are modeled with a deterministic drift component $\mu$ and a stochastic Brownian component with variance $\sigma^2$ : 
$$\frac{dX}{X} = \mu dt + \sigma dB$$

Using \ref{eq:ito2} we note:

$$
\begin{align*}
    df(t,X)&= \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial X}\left( \mu Xdt + \sigma XdB\right) + \frac{1}{2} \frac{\partial^2 f}{\partial X^2} \left(\mu  Xdt + \sigma XdB\right)^2 \\
    &=\left(\frac{\partial f}{\partial t} + \mu X\frac{\partial f}{\partial X} + \frac{1}{2} {\sigma}^2 X^2\frac{\partial^2 f}{\partial X^2}\right)dt + \sigma X\frac{\partial f}{\partial X}dB
\end{align*}
$$

##### Example: Product and Quotient rule
We can use the above theory and try to compute dervative for $f(X,Y)=XY$ where both $X$ and $Y$ are stochastic processes. We already know for non stochastic variables, this is the product rule:

$$
d(XY) = XdY + YdX
$$

For stochastic process, considering all second order terms:

$$
\begin{align*}
d(XY) &= \frac{\partial f}{\partial X}dX + \frac{\partial f}{\partial Y}dY + \cancelto{0}{\frac{1}{2} \frac{\partial^2 f}{\partial X^2}dX^2} +  \cancelto{0}{\frac{1}{2} \frac{\partial^2 f}{\partial Y^2}dY^2} + \frac{\partial^2 f}{\partial X \partial Y}dXdY\\
&= YdX + XdY + dXdY
\end{align*}
$$

Similarly for $f(X,Y) = \frac{X}{Y}$:

$$
\begin{align*}
d(XY) &= \frac{\partial f}{\partial X}dX + \frac{\partial f}{\partial Y}dY + \cancelto{0}{\frac{1}{2} \frac{\partial^2 f}{\partial X^2}dX^2} +  \frac{1}{2} \frac{\partial^2 f}{\partial Y^2}dY^2 +  \frac{\partial^2 f}{\partial X \partial Y}dXdY\\
&= \frac{1}{Y}dX - \frac{X}{Y^2}dY + \frac{X}{Y^3} dY^2 - \frac{1}{Y^2}dXdY
\end{align*}
$$

<!--##### Application to option pricing (will come back to this)
Lets use our understanding to further apply to derivatives of european call option price. 
-->

### Conclusion
That's it! These are the main results to know in stochastic calculus to be able to understand some of the derivatives pricing equations like BSM. There is endless more maths involved but I think understanding the above should be enough to compute on our own some of the arithmetic. Some references for above material are: {% cite mit_financial_maths %} and {% cite quantpie %}