---
layout: post
title: "Understanding the Attention Mechanism in Transformers"
date: 2025-10-28 22:54:00
categories: [deep-learning, transformers]
tags: [attention, transformers, neural-networks]
---

At an API level, an **attention layer** is a sequence to sequence model that takes a sequence of input vectors and produces a new, modified sequence of output vectors of the same or different dimension. 

This note builds up the attention mechanism step by step, starting from the simplest case and gradually adding complexity. We will proceed as follows:

1. Attention with a single query vector  
2. Attention with a sequence of query vectors  
3. Attention with learnable parameters

 An attention layer receives a sequence of **query** vectors, a sequence of **key** vectors and a corresponding set of **value** vectors (same number as keys). Attention layer behaves like a fuzzy dictionary, finding keys closest to the query and returning the value corresponding to the keys. The final returned value is weighted sum of all given values with weights being equal to the similarity between the corresponding key and query. We will show mathematically how this is done. 

#### 1. Attention with a Single Query Vector
In the first setting, we consider a case where the sequence of query vectors has only a single query vector. This is represented by matrix $Q_{1\times d}$. We have $N_k$ keys represented by rows of matrix $K_{N_k \times d}$. Number of value vectors are equal to number of keys $N_k$, as corresponding to each key there is a single value.

Similarity matrix is computed as: 

$$
S_{1\times N_k} = Softmax(Q_{1\times d}K_{N_k\times d}^T)
$$

Finally, to get the output value vector we will take a weighted sum of $N_k$ "value" vectors, where the weights are based on the similarity scores in S. The scores are normalized by applying softmax operation across each **row** of matrix $QK^T$. The softmax ensures that all weights are positive and sum to one. Note that we need each query and key to be in the same dimension space to allow for dot product similarity.

$$
Attention(Q,K,V)_{1\times d} = S_{1\times N_k}V_{N_k\times d}
$$

###### Computational Complexity

1. Computation of matrix $QK^T$ takes $N_k\times d$ multiplication operations
2. Normalization through softmax takes $N_k$ steps, considdering $exp()$ operations takes$O(1)$ time with small constant as it is hardware accelerated on GPU and on CPU fast exponentiation libraries like in math.h in C++ exist.
3. Finally computing weighted sum of "values" will take $N_k\times d$ multiplications.  

$$
{\text{Time complexity: } O(N_k d)}
$$


#### 2. Attention with a Sequence of Query Vectors
Now lets consider slightly more complicated setting where we have $N_q$ queries, each represented by a row in matrix $Q_{N_q \times d}$. Similar to single query scenario, we compute dot product similarity between each query and key as 

$$
S_{N_q \times N_k} = softmax(QK^T)
$$

Each row of S represents similarity score of that query with all the $N_k$ keys. Finally multiply S with value matrix $V_{N_k \times d}$ to get $N_q$ output values. 

$$
Attention(Q,K,V)_{N_q\times d} = S_{N_q \times N_k} V_{N_k \times d}
$$

$$
\text{Time complexity}: O(N_q \times N_k \times d)
$$

This is assuming that "query" vector dimension and "value" vector dimension are the same. However, this may not always be true. If "value" vector has dimension d_v: 

$$
\text{Time complexity}:O(N_q \times N_k \times d + N_q \times N_k \times d_v)
$$

In **self-attention**, same sequence of 
vectors are passed as key, query and values i.e. $N_q = N_v = N_k = N$ and we assume output value vectors to be in same dimension i.e. $d_v = d$. Thus, we get 

$$\boxed{\text{Time complexity}: O(N^2*d)}$$ 


##### Scaling Softmax
There is also a scaling nuance we need to address before moving further. Let's focus on similarity matrix $S$. Suppose each entry of query Q and key K is an independent random variable with $\mu=0$, $\sigma^2 = 1$. Then, 

$$
\begin{align*}E[s_{ij}] &= E[\mathbf{q}_i \cdot \mathbf{k}_j] = E\left[\sum_t q_{it}k_{jt}\right] = \sum_t E[q_{it}k_{jt}] \text{ (by linearity of expectation)} \\
&= \sum_t E[q_{it}]E[k_{jt}] \text{ (independence of random variables) } \\
&= 0
\end{align*}
$$

So, expected value of each term before softmax operation is 0. Let's also check its variance. 

$$
\begin{align*}
\operatorname{Var}[s_{ij}] &= \operatorname{Var}\left[\sum_t q_{it} k_{jt}\right] \\
&= \sum_t \operatorname{Var}[q_{it} k_{jt}] \text{ (since each of summand is independent)}\\
&= \sum \operatorname{Var}[q_{it}] \operatorname{Var}[k_{jt}] \\
&= d
\end{align*}
$$

Last equality follows since $Var(XY) = E[X^2Y^2]- (E[XY])^2 = E[X^2]E[Y^2] = Var(X)Var(Y)$

This shows that each $s_{ij}$ has mean $0$ but variance is large ($d$). As we do softmax, for large values of $s_{ij}$, softmax saturates i.e. the derivative of softmax vanishes leading to learning being stopped in gradient descent. To avoid this we need to prevent $s_{ij}$ from getting too large. We scale $s_{ij}$ by $\frac{1}{\sqrt d}$ as $var[s_{ij}/\sqrt(d)] = var[s_{ij}]/d = 1$ 


#### 3. Attention with learnable weights parameters
 So far the attention block has no learnable parameters, it is simply a fuzzy dictionary. As next layer of complexity we linearly transform the queries, keys and values into another vector space of different dimension before computing similarity matrix. 
 1. Keys are linearly transformed as 
   
   $$
   \hat{K}_{N_k \times d_k} = K_{N_k \times d}*W^K_{d \times d_k} 
   $$

   2. Queries are in the same dimension space as keys 
   
   $$
   \hat{Q}_{N_q \times d_k} = Q_{N_q \times d}*W^Q_{d \times d_k}
   $$
   
   3. Values are transformed as 
   
   $$
   \hat{V}_{N_k X d_v} = V_{N_k \times d}*W^V_{d \times d_v} 
   $$ 
   
   Again we do the same similarity matrix computation followed by softmax and multiplication by V_hat to get attention output sequence. 
   
   $$
   \boxed{Attention(Q,K,V)_{N_q\times d} = softmax(\frac{\hat{Q}\hat{K}^T}{\sqrt{d}})\hat{V}}
   $$
   
   Time Complexity is $O(N_k \times N_q \times d_k + N_k \times N_q \times d_v + N_k \times d_k \times d + N_q \times d_k \times d + N_k \times d_v \times d)$. 
   
   For self attention, assuming output sequence vector is of dimension $d_v = d$: 
   
   $$
   O(N^2d + Nd^2 )
   $$
   
   since $N >> d$ (long context window) self attention is practically $O(N^2d)$. The qudratic complexity in context length makes attention expensive for long sequences. 
   
   ##### How many learnable parameters does attention layer has? 
   Considering keys, queries and values are all transformed to same dimension $d$ equal to the input dimension, total learnable parameters: $3d^2$. Note, learnable parameters matrix doesn't depend on length of input sequence. This makes intuitive sense, as each training example can be of different sequence length (each input sentence can have different number of words/token) and still we can update parameters of the model.

---

## Closing Remarks

Further extensions — **multi-head attention**, **cross-attention**, and **masked attention** — expand this mechanism to capture richer relationships. We will explore these in future notes.