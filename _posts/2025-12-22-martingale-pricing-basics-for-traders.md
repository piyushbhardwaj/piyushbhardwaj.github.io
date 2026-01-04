---
layout: post
title: "Martingale Pricing Basics for Traders"
date: 2025-12-22 12:00:00
categories: [quantitative-finance, martingale-pricing]
tags: [martingale, risk-neutral, replication, no-arbitrage, gbm, measure-change]
giscus_comments: true
related_posts: false
related_publications: true
---

Certain [stochastic processes]({% post_url 2025-12-07-stochastic-calculus-basics-for-traders %}) can be classified as Martingales. It is useful to understand martingale processes since they form the basis of derivative pricing. We will start with an intutive definition of martingale and then look at few stochastic processes that are martingales. 

### Martingale
A Martingale is a stochastic process where expected value of next step given all past information upto that time is equal to its present value i.e.

$$
E\left[X_{t+1}\mid F_t\right] = X_t
$$

where $F_t$ can be understood as collection of all information upto time $t$. Intutively, the above expectation can be thought of as the best possible guess for the value at next step, and for martingale processes it turns out the best guess is the last known value. In a more general sense, it follows from above (using [tower property of expectation](#tower-property-of-expectation)):

$$
E\left[X_{t+1}\mid F_s\right] = X_s,\; \forall\; s\le t
$$

Above property is useful in pricing derivatives, if an option price process is a martingale its value $X_s$ at $t=s$ can be computed as expected value at expiry which is equal to option's payoff. However, before going to answer the question of option price being a martingale, lets check few stochastic processes.

##### 1. Gambler's wealth
Let $X_t$ represent a gambler's wealth at time $t$. At each time step he plays a game where his wealth goes up or down by $1$ with equal probability. 

$$
E[X_{t+1}|F_t] = E[X_t+\epsilon|F_t] = E[X_t|F_t] + E[\epsilon|F_t] = X_t
$$

That makes sense, since at each turn on an average he makes no money and wealth remains same as previous step. In context of trading I often hear the term martingale betting which means doubling your betsize if you lose in a fair game. The intuition is that if I keep on doubling my bet at every time step, eventually when I win, I will recover all my losses and win some more. Eventuality of winning is guaranteed as probability of losing $N$ games consecutively goes to zero ($2^{-N}$) as $N\to\infty$. The fallacy here is that for a fair game the time to win for the first time ($\tau$ which is a random variable) can be infinite i.e. gambler will win eventually but $E[\tau] = \infty$.

##### 2. Brownian Motion
Let $B(t)$ denote a [brownian motion]({% post_url 2025-12-07-stochastic-calculus-basics-for-traders %}/#brownian-motion) process. Let's compute:

$$
E[B_t|F_s] = E[B_s + (B_t - B_s)|F_s] = E[B_s|F_s] + E[(B_t - B_s)|F_s] = B_s + 0 = B_s
$$

where the second term $E[(B_t - B_s)\|F_s] = 0$ since by independent increments $B_t - B_s$ is independent of $F_s$ and $B_t-B_s \sim N(0, t-s)$ making expectation $0$. Hence, brownian motion is a martingale.  

###### 2.1 Brownian Motion with drift
Next, lets look at brownian motion motion with drift i.e. $X_t = \mu t + B_t$. This is not a martingale as the conditional expectation grows with time, i.e. 

$$
\begin{align*}
E[X_t|F_s] &= E[X_s + (X_t - X_s)|F_s] = E[X_s|F_s] + E[(X_t - X_s)|F_s] \\
&= X_s +  \mu (t-s) + E[(B_t - B_s)|F_s] = X_s +  \mu (t-s)
\end{align*}
$$

###### 2.2 Geometric Brownian Motion without drift 
Let $X_t$ be a GBM stochastic process defined as 

$$
\frac{dX}{X} = \sigma\,dB
$$

From [equation]({% post_url 2025-12-07-stochastic-calculus-basics-for-traders %}#eq-log-returns) in previous post, we already know the solution of the SDE as:

$$
X_t = X_0 \exp\left(- \frac{\sigma^2t}{2} + \sigma B(t)\right)
$$

This can be written in terms of $X_s$ as:

$$
\begin{align*}
X_t &= X_0 \exp\left(- \frac{\sigma^2(s+(t-s))}{2} + \sigma B(s)+(B(t)-B(s))\right) \\
&= X_0\exp\left(- \frac{\sigma^2s}{2} + \sigma B(s)\right)\exp\left(- \frac{\sigma^2(t-s)}{2} + \sigma (B(t)-B(s))\right)\\
&= X_s\exp\left(- \frac{\sigma^2(t-s)}{2} + \sigma (B(t)-B(s))\right)
\end{align*}
$$

Taking expectation on both sides:

$$
E[X_t|F_s] = E\left[X_s\exp\left(- \frac{\sigma^2(t-s)}{2} + \sigma (B(t)-B(s))\right)|F_s\right]
$$

Since $X_s\|F_s$ is deterministic, it can be taken out of expectation. Since $B(t) - B(s)$ is independent of $F_s$, conditional expectation can be dropped. 

$$
\begin{align*}
E[X_t|F_s] &= X_s\exp\left(- \frac{\sigma^2(t-s)}{2}\right)E\left[e^{\sigma (B(t)-B(s))}\right]\\
&= X_s
\end{align*}
$$

This follows from $E\left[e^{\sigma (B(t)-B(s))}\right] = e^{\frac{\sigma^2(t-s)}{2}}$, which in turn can be shown using moment generating function for normal distribution. MGF for $X\sim N(0,1)$: 

$$
E[e^{tX}] = e^{t^2/2}
$$

This shows that GBM without drift term is a martingale. However, stock price process defined by below GBM with drift is not a martingale.

$$
\frac{dS}{S} = \mu\,dt + \sigma\,dB
$$

This is not surprising. In the long run we expect stocks to grow. The growth rate above the risk-free rate is the equity risk premia i.e. the risk one takes to accept the variance in stock prices.

Can we make some kind of transformations to the GBM process to make the stock price a martingale. If we can put some handle on the drift term, we can perhaps get to a martingale process. The path to that goes through the abstract maths of measure space. 

#### Change of Measure
Simply put a probability measure can be thought of as assignment of probabilites to different elements of the sample space $\Omega$, such that they add up to 1. For instance, when we roll a dice such that each face has probability $\frac{1}{6}$ of turning up, we can say that in this measure $P(X=i) = \frac{1}{6}$. We can think of another assignments of these probabilities such that sum of probability assignment is $1$, and that will be another valid probability measure. When we change the measure, we have gone from a fair dice to a loaded dice. However, the value of random variable remains the same, i.e. it takes values from the same set, just the probability of these values are reassigned.

In terms of a stock price process, we may think of changing measure as assigning different probability to different stock price path. As we discussed for dice, this new probability assignment may be very different from the actual physical reality (physical measure). Brownian motion in the actual physical measure, mayn't be brownian motion in the transformed measure. However, this new imaginary probability measure could be simpler to model. What if we can find an imaginary measure in which stock price process is indeed a martingale. Even if that happens, two open questions remain:
1. Why would an imaginary measure help in pricing assets in real world?
2. Can such a measure even exsist? 

##### Radon-Nikodym derivative
Let's answer the second question first. In measure theory there is a tool known as Radon-Nikodym derivative which helps transform between two probability measures given these probability measures are equivalent. Two measures are called equivalent when all events assigned non-zero probability in one space are also assigned non-zero probability in the other space and vice-versa. Intutively, changing measure shouldn't invent new paths. 

Consider two equivalent measures $P$ and $Qs$, we can convert all points $w$ in the sample space from one measure to another by using a change of measure function $z(w)$ (called Radon-Nikodym derivative). Consider an event $A$, we define the probability of event occuring as

$$
P(A) = \int_{w \in A} dp(w)
$$

We can assign new probabilities to event $A$ under the new measure $Q$ as

$$
Q(A) = \int_{w \in A} dq(w) = \int_{w \in A} z(w)\, dp(w)
$$

The function $z$ is basically redistributing probabilities across points in the sample space. It is called a derivative since it is since it can be written in terms of densities as $\frac{dQ}{dP}$

##### Transforming to Standard Normal
Consider a random variable $X$ which follows standard normal distribution $N(0,1)$ in measure $P$. Let $Y = X + \theta$, where $\theta$ is a constant. Under measure $P$, $Y \sim N(\theta, 1)$. Does there exist a measure $Q$ in which $Y \sim N(0,1)$? 

We claim that by choosing $z(w) = e^{-\theta X(w) - \tfrac{1}{2} \theta^2}$, the equivalent measure has $Y\sim N(0,1)$. Note under measure $P$

$$
P(X \in [a,b]) = \int_a^b \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx
$$

Using Radon-Nikodym, under measure $Q$ we get

$$
\begin{align*}
Q(X \in [a,b]) &= \int_a^b z(x) \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx \\
&= \int_a^b e^{\{-\theta X - \tfrac{1}{2} \theta^2\}} \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx \\
&= \int_a^b \frac{1}{\sqrt{2\pi}} e^{-(x+\theta)^2/2} dx \\
Q(X+\theta \in [a+\theta,b+\theta]) &= \int_a^b \frac{1}{\sqrt{2\pi}} e^{-(x+\theta)^2/2} dx
\end{align*}
$$

Replacing $y = x+\theta$, we get the result that $Y \sim N(0,1)$ in $Q$.

$$
\begin{align*}
Q(Y \in [a+\theta,b+\theta]) &= \int_{a+\theta}^{b+\theta} \frac{1}{\sqrt{2\pi}} e^{-y^2/2} dy
\end{align*}
$$

##### Grisonov's Theorem
This can be further extended to Brownian motion. Let $B(t)$ be Brownian in the physical measure $P$ i.e. $B(t) \sim N(0,t)$. Consider a stochstic process $X(t) = B(t) + \theta(t)$, where $\theta(t)$ is a non-stochastic function of time. Under measure $P$ $X(t)$ is brownian with drift. It can be shown that there exist an equivalent measure $Q$ in which $X(t)$ is Brownian without drift. This is the simple form of famous Grisonov's theorem. The transformation function $z(w)$ is very similar to what we have seen above. 

#### Risk Neutral Measure
We can use the Grisonov's theorem to show that we can ingeniously choose $\theta(t)$ such that the discounted stock price process is a martingle. Let $\theta(t) = \frac{\mu - r}{\sigma} t$, then $B(t)+\theta(t)$ is brownian in measure $Q$. We define this as $B^Q$:

$$
B^Q(t) = B(t) + \frac{\mu - r}{\sigma}\, t
$$

If we replace this in the equation of GBM price process, we get the price process in this imaginary world as:
$$
\begin{align}
\label{eq:gbm2}
\frac{dS}{S} = r\,dt + \sigma\, dB^Q
\end{align}
$$

We note that under the above stock price process, the discounted stock price $Se^{-rt}$ is drift less (hence a martingale)

$$
\begin{align*}
d(Se^{-rt}) &= -rSe^{-rt}dt + e^{-rt} dS \\
&= -rSe^{-rt}dt + e^{-rt}(rSdt + \sigma S\, dB^Q) \\
&= Se^{-rt}\,\sigma\, dB^Q \\
\frac{d(Se^{-rt})}{Se^{-rt}} &= \sigma\, dB^Q
\end{align*}
$$

Note, equation \ref{eq:gbm2} is not a martingale because it is not driftless (note we saw that [GBM without drift is a martingale](#22-geometric-brownian-motion-without-drift)). Yet, this imaginary world is very interesting because instead of drift $\mu$, stocks in this world grow at risk free rate $r$, that's why this is also known as risk-neutral measure.

This resolves one of the two questions that we raised. There indeed exsists an equivalent measure Q, in which the discounted stock price process is martingale. We now move to the other question: Why would an imaginary measure help in pricing assets in real world?

Let's start by noting that had we chosen $B^Q(t) = B(t) + \frac{\mu}{\sigma}t$, $B^Q(t)$ would still have been brownian and the stock price process without discounting would also have been a martingale. So, why did we go the discounted route and not the simpler route. The answer lies in the fact that this imaginary world where stock price is a martingale will not be arbitrage free. 

#### Fundamental theorem of asset pricing (FTAP)
An arbitrage roughly can be understood as making profit from nothing under all possible market scenarios. Under efficient market hyptohesis, markets should be arbitrage free, there is no free lunch. For arbitrage free markets, FTAP says that there exists an equivalent measure in which all freely tradable assets' discounted price procees are martingales. The inverse is also true, if there exists an equivalent martingale measure for all tradable assets then the market is arbitrage free. By FTAP, every asset price can be written as it's expected value at $T$ in $Q$-measure. We have seen in the previous section that we exactly understand the Q measure under which discounted prices are martingale. This makes pricing much simpler, we can price call option at time $t$ as:

$$
\begin{align*}
C(K, t) = e^{-r(T-t)}E^Q[max(S_T-K, 0)]
\end{align*}
$$

Given we know $Q$, it's a matter of integration for computing this expectation.

#### Hedging and Replicating Portfolio
Now that we know how to compute price of any derivative, we will now address the question: can we hedge derivatives using primitive securities like stocks and cash. We can hedge if we can create a portfolio of primitive securities that can track the price of the derivative under all market scenarios. Such a portfolio is called a replicating portfolio. 


<!--#### Self financing Portfolio

#### Complete markets

### Replication and no-arbitrage -->

A derivative is said to be replicable if we can create a portfolio of stock and cash (primitive securities) such that:

1. Value of portfolio at maturity is equal to the value of derivative under all possible paths
2. Portfolio is self-financing i.e. there is no inflow or outflow of cash from the portfolio.

As the value of the derivative at maturity is equal to value of portfolio and since the portfolio is self-financing, the value of portfolio and derivative should be same at all times irrespective of the path of the stock, unless there is arbitrage in the market. Under the no-arbitrage assumption, the cost of setting up such a portfolio is the value of the derivative. <!--Note that the replication has nothing to do with the probability measure we choose. Choosing a different measure only changes the assignment of probabilities to different paths but because the measures are equivalent all non-zero probability paths are consistent across measures. This means the value of replicating portfolio should be same irrespective of the measure chosen and as value of the replicating portfolio is same as the value of derivative, valuations in different measures should be the same (as long as there is no-arbitrage condition).-->

<!--To summarize, if there is replicability, we can setup a portfolio paying same as the derivative at expiry and as no-arbitrage holds, this portfolio is valued same as the derivative at all earlier times. As the above argument has nothing to do with the measure under consideration, pricing under all equivalent measures should be the same. Finally, as pricing under martingale measure is simpler, we choose that over other measures. -->

If the stock process is martingale in $Q$-measure, does that make derivative process also a martingale. The answer is yes, and it can be understood through the replication argument. Let's say a derivative $C$ is replicated using a portfolio $\Pi$ of $\delta_0$ cash and $\delta_1$ stock at time $t=0$.

$$
\Pi(0) = \delta_0 + \delta_1 S(0)
$$

At next step $t=1$, sincce this is a replicating portfolio, its value is equal to the value of derivative $C(1)$:

$$
\Pi(1) = \delta_0 e^{r} + \delta_1 S(1) = C(1)
$$

Taking $e^r$ on the other side and taking expectation wrt to martingale measure $Q$:

$$
\begin{align*}
E^Q[\Pi(1)e^{-r}] &= \delta_0 + \delta_1 E^Q[S(1) e^{-r}] = E^Q[C(1) e^{-r}] \\
E^Q[C(1) e^{-r}] &= \delta_0 + \delta_1 S(0) = \Pi(0)
\end{align*}
$$

As under no-arbitrage $\Pi(0) = C(0)$, we can see that discounted derivative price is also a martingale. Thus, it is enough to find a measure in which stock is martingale and the derivative will follow with replication and no-arbitrage.

Finally we summarize the two well-known fundamental theorems of asset pricing (FTAP). They show why we can always do valuation of all traded assets under the equivalent martingale measure (Q) given no-arbitrage and replicability holds.

1. FTAP1: No arbitrage model implies existence of at least one equivalent martingale measure. Existence of at least one equivalent martingale measure implies no-arbitrage.
2. FTAP2: If the market is complete i.e. each and every claim can be replicated by primitive securities and there is no-arbitrage then there exist exactly one equivalent martingale measure. This means the price of every derivative is unique and that is the cost of replication which is same as discounted expected value of payoff at expiry under risk-neutral measure.

#### Tower Property of Expectation
TBD

Some references for above material are:  {% cite mit_financial_maths %}, {% cite finmath_simplified %}, {% cite caltech_BEM1105x %}, {% cite RangaraSundaram %}, {% cite uchicago_statistics390 %}

