---
layout: post
title: "Understanding the Attention Mechanism in Transformers"
date: 2025-10-28 22:54:00
categories: [deep-learning, transformers]
tags: [attention, transformers, neural-networks]
giscus_comments: true
---

At an API level, a transformer block can be thought of as a sequence-to-sequence model that takes a sequence of vectors as input and produces a new, modified sequence of vectors as the output. Transformer block has an attention layer and a feed-forward layer. The attention layer in itself is also a sequence-to-sequence layer. In this note we will focus on the attention layer, starting from a simple case and gradually building complexity. We will proceed as follows:

1. Attention with a single query vector  
2. Attention with a sequence of query vectors  
3. Attention with learnable parameters
4. Multi-head Attention

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-2 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/transformer_block.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Transformer block with attention and feed-forward layer
</div>

 An attention layer receives a sequence of **query** vectors and a sequence of **key-value** vector pairs. Attention layer behaves like a fuzzy dictionary, finding keys closest to the query and returning the value corresponding to the keys. The final returned value is weighted sum of all given values with weight as similarity between key and query. We will show the algebra of how this is done. 

#### 1. Attention with a Single Query Vector
In the first setting, we consider a case where the sequence of query vectors has only a single query vector. This is represented by matrix $Q_{1\times d}$. We have $N_k$ keys represented by rows of matrix $K_{N_k \times d}$. Number of value vectors are equal to number of keys $N_k$, as corresponding to each key there is a single value. The similarity matrix is computed as: 

$$
S_{1\times N_k} = Softmax(Q_{1\times d}K_{N_k\times d}^T)
$$

Finally, to get the output value vector we will take a weighted sum of $N_k$ "value" vectors, where the weights are based on the similarity scores in S. The scores are normalized by applying softmax operation across each **row** of matrix $QK^T$. The softmax ensures that all weights are positive and sum to one. Note that, we need each query and key to be in the same dimension space to allow for dot product similarity.

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

$$\boxed{\text{Time complexity}: O(N^2d)}$$ 


##### Scaling Softmax
There is also a scaling nuance we need to address before moving further. Let's focus on similarity matrix $S$. Suppose each entry of query Q and key K is an independent random variable with $\mu=0$, $\sigma^2 = 1$. Then, 

$$
\begin{align*}E[s_{ij}] &= E[\mathbf{q}_i \cdot \mathbf{k}_j] \\
                        &= E\left[\sum_t q_{it}k_{jt}\right] \\
                        &= \sum_t E[q_{it}k_{jt}] \text{ (by linearity of expectation)} \\
                        &= \sum_t E[q_{it}]E[k_{jt}] \text{ (independence of random variables) } \\
                        &= 0
\end{align*}
$$

So, expected value of each term before softmax operation is 0. Let's also check its variance. 

$$
\begin{align*}
\operatorname{Var}[s_{ij}] &= \operatorname{Var}\left[\sum_t q_{it} k_{jt}\right] \\
&= \sum_d \operatorname{Var}[q_{it} k_{jt}] \text{ (since each of summand is independent)}\\
&= \sum \operatorname{Var}[q_{it}] \operatorname{Var}[k_{jt}] \\
&= d
\end{align*}
$$

Last equality follows since 
$$
\begin{align*}
Var(XY) &= E[X^2Y^2]- (E[XY])^2 \\ 
&= E[X^2]E[Y^2] \\
&= Var(X)Var(Y)
\end{align*}
$$

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

#### How many learnable parameters does attention layer has? 
Considering keys, queries and values are all transformed to same dimension $d$ equal to the input dimension, total learnable parameters: $3d^2$. Note, learnable parameters matrix doesn't depend on length of input sequence. This makes intuitive sense, as each training example can be of different sequence length (each input sentence can have different number of words/token) and still we can update parameters of the model.

#### Intuition of attention
The output values are supposed to capture the contextual information from other tokens in the sentence. The softmax weights dictate the importance of each token in the output vector for a given input token. However, the softmax matrix is just a dot product similarity matrix. Why should that be equivalent to contextual information? It is hard to put this down in a small blog post and perhaps needs another blog post of its own. However, to put it succinctly, the magic is happening in the weights matrices $W_Q$, $W_K$ and $W_V$. Each matrix linearly transforms the input vector into another vector space where it's meaning is transformed from itself to "some contextual aspect". What the "contextual aspect" is going to be is left upto the network to learn by adapting the weight matrices. It would be interesting to see post the training finishes what these matrices turn out to be and what they actually mean in different learning tasks. Often instead of learning a single large $W_Q$, $W_K$ and $W_V$, we learn multiple such matrices (called attention heads) and contenate the result of each head to form the final output vector. 

### 4. Multi-head Attention

In multi-head attention we transform each key, query and value matrix into $H$ different sub-spaces by using $H$ different weight matrices. For instance now we will have $H$ different weight matrices $W^K_1, W^K_2,\ldots, W^K_H$ for transforming key matrix $K$. We are dropping the notation for matrix size in this section for ease of reading.  

1. Keys are linearly transformed as 
   
   $$
   \hat{K}_i= KW^K_i
   $$

2. Queries are in the same dimension space as keys 
   
   $$
   \hat{Q}_i = QW^Q_i
   $$
   
3. Values are transformed as 
   
   $$
   \hat{V}_i = VW^V_i
   $$ 

Each of these "heads" will produce its own attention mechanism and produce output matrix $head_i$ for $i \in [1, H]$

$$
head_i = softmax(\frac{\hat{Q}_i\hat{K}_i^T}{\sqrt{d}})\hat{V}_i
$$

As each of the heads is of dimension $N_q \times d_v$, they can all be stacked horizontally to produce a concatenated matrix of dimenstion $N_q \times (Hd_v)$. 
   
   $$
   Attention(Q,K,V)_{N_q\times Hd_v} = concat(head_1, \ldots, head_H)
   $$

Sidenote: What just happened? The idea of concatenating feels wierd and so non-linear. Inutively, this step is just concatenating representations learnt through different mechanisms enforced through the weight matrices. Each of the head is contributing a different "contextual" meaning to the output vector for the query. Say, one head is trying to understand part of speech and another is trying to understand token as some particualr entity. This is just how i think about it. If I find literature around it I will link here. 

#### Did we increase number of parameters by adding more heads?
Assuming query, key and value are transformed to the same dimension $d_h$, total number of parameters in the attention layer are $3Hdd_h$. Typically, we choose $d_h = \frac{d}{H}$. This makes total number of parameters as $3d^2$, same as for transformer without multihead attention. The choice of $d_h = \frac{d}{H}$ is arbitrary, we could have chosen any other dimension as well. Transformers are ofthen stacked one on another i.e. output of one transformer layer goes as input to the next transformer layer, so it is simpler if each component of the stack has same output shape as input shape. However, this doesn't necessiate that we choose $d_h = \frac{d}{H}$ as in the architecture there is another set of weights $W_O$ multiplied to the output of attention space to transform the concatenated attended vectors to desired output dimension.  

$$
   O_{N_q \times d} = concat(head_1, \ldots, head_H) W^O
$$


#### Did we increase number of multiplications by adding more heads?
Let's count! Assuming $N_q = N_v = N_k = N$, for each head we will be doing $N \times d \times d_h$. For all heads combined we will have $Nd^2$ multiplications. We can parallelize computation of each of the head. However, as far as parallelization goes, a **lot** can be parallelized. 

---

### Closing Remarks

Further extensions — **cross-attention**, and **masked attention** — expand this mechanism to capture richer relationships. We will explore these in future notes.