---
title: "Stochastic Calculus Basics for Traders"
date: 2024-04-13
layout: single
author_profile: true
tags: [Stochastic Calculus, Finance, Brownian Motion, Ito, GBM]
math: true
---

This blog post summarizes fundamental concepts in **stochastic calculus**, particularly as they relate to modeling stock prices and derivative pricing.

---

## 1. Stochastic Process

We start by modeling the stock price as a time-dependent random variable, i.e., a **stochastic process**. This is essentially a sequence of random variables indexed by time (continuous in this case).

We assume:
- Price change over short intervals is normally distributed with mean 0 and variance Δt
- Changes over non-overlapping intervals are independent

This simplifies real-world behavior and helps build mathematical intuition.

---

## 1.1 Brownian Motion

**Brownian motion** (also called Wiener process) satisfies:

- $B(0) = 0$
- $B(t) - B(s) \sim \mathcal{N}(0, t - s)$
- Increments over non-overlapping intervals are independent

---

## 1.2 Properties of Brownian Motion

### Non-differentiability

$B(t)$ is not differentiable:  
$\lim_{h \to 0} \frac{B(t + h) - B(t)}{h}$ does **not** exist.  
Despite appearing smooth, Brownian paths are too erratic to define a slope.

### Quadratic Variation

For a well-behaved function $f(x)$:

$$
\sum (f(t_{i+1}) - f(t_i))^2 \to 0
$$

But for Brownian motion:

$$
\sum (B(t_{i+1}) - B(t_i))^2 = T
$$

i.e., $\Delta B^2 = \Delta t$

---

## 1.3 Ito's Calculus

Because $B(t)$ isn't differentiable, we can't apply standard calculus. Instead, we use **Ito's calculus** to analyze functions of stochastic processes.

### Taylor Expansion Analogy

Regular calculus:

$$
\Delta f(x) = f'(x)\Delta x
$$

Stochastic version:

$$
\Delta f(B_t) = f'(B_t)\Delta B_t + \frac{1}{2}f''(B_t)\Delta t
$$

As $\Delta B_t \sim \mathcal{N}(0, \Delta t)$, we treat $\Delta B_t^2 = \Delta t$.

Thus, Ito's formula becomes:

$$
df = f'(B)\,dB + \frac{1}{2}f''(B)\,dt
$$

---

### 1.3.1 Ito's Lemma for $f(t, X)$

If $X$ is stochastic and $t$ is deterministic:

$$
df(t, X) = \frac{\partial f}{\partial t}\,dt + \frac{\partial f}{\partial X}\,dX + \frac{1}{2}\frac{\partial^2 f}{\partial X^2}\,dX^2
$$

If $dX = dB$, then $dX^2 = dt$, and we get:

$$
df = \left(\frac{\partial f}{\partial t} + \frac{1}{2}\frac{\partial^2 f}{\partial X^2}\right)dt + \frac{\partial f}{\partial X}dB
$$

---

### 1.3.2 Application to Geometric Brownian Motion (GBM)

Stock price modeled as:

$$
\frac{dX}{X} = \mu\,dt + \sigma\,dB
$$

Applying Ito's lemma:

$$
df(t, X) = \left(\frac{\partial f}{\partial t} + \mu X \frac{\partial f}{\partial X} + \frac{1}{2}\sigma^2 X^2 \frac{\partial^2 f}{\partial X^2}\right)dt + \sigma X \frac{\partial f}{\partial X}dB
$$

---

## 1.4 Conclusion

These results lay the foundation for pricing models like Black-Scholes. Ito's product and quotient rules extend regular calculus:

- Product:
  $$
  d(XY) = X\,dY + Y\,dX + dX\,dY
  $$
- Quotient:
  $$
  d\left(\frac{X}{Y}\right) = \frac{1}{Y}dX - \frac{X}{Y^2}dY + \frac{X}{Y^3}dY^2 - \frac{1}{Y^2}dX dY
  $$

---

## 📚 References

- MIT OCW: [Topics in Mathematics with Applications in Finance](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/resources/mit18_s096f13_lecnote18/)
- [QuantPie](https://quantpie.co.uk/)
