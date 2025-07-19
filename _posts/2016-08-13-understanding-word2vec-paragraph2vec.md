---
title: "Understanding Word2Vec and Paragraph2Vec"
date: 2016-08-13
layout: single
author_profile: true
tags: [Word2Vec, Paragraph2Vec, NLP, Embeddings]
---

This post summarizes my notes on understanding the update steps for the Para2Vec (Paragraph2Vec) algorithm, particularly focusing on the **Distributed Memory model (DM-mean)** — a close extension of the Word2Vec algorithm.

---

## 1. Word2Vec Architecture

We focus on the **Continuous Bag of Words (CBOW)** model with negative sampling, using mean at the hidden layer. This model consists of a single hidden layer.

Let:

- **WI** and **WO** be the input and output vocabulary sets.
- **pi**, **qi** ∈ ℝ<sup>r</sup> be the input/output vectors.
- **P** and **Q** are the input/output weight matrices.

### Hidden Layer Representation

The hidden layer vector **h** is the mean of input vectors:

```
h = (1 / |C|) * Σ pi  where i ∈ C
```

### Output Layer Prediction (Softmax)

The softmax probability of the target word:

```
P(w*_O | C) = exp(q*_jᵗ h) / Σ exp(q_jᵗ h)
```

### Output Vector Update

```
qᵢ ← qᵢ + η * (I(i = j*) - P(wᵢ | C)) * h
```

### Input Vector Update

```
pᵢ ← pᵢ + η / |C| * Σ (error_j * qⱼ)
```

---

## 2. Negative Sampling

To avoid softmax cost, we use **negative sampling** with the logistic (sigmoid) function:

```
σ(x) = 1 / (1 + e^(-x))
```

Objective becomes:

```
log σ(q*_jᵗ h) + Σ log σ(-q_jᵗ h) for j ∈ Negative Samples
```

The gradient update for output and input vectors remains similar, with updates restricted to the target and sampled negatives.

---

## 3. Paragraph2Vec (Doc2Vec)

The key idea: **documents are treated as additional context vectors**, much like words.

During training:
- A document vector is used as part of the context set C.
- The same update rules apply, with doc vectors in **P**.

---

## 4. Inference for New Documents

To infer a document vector **d_test**, fix the word vectors and only update **d_test** using:

```
d_test ← d_test + η / |C| * Σ (error_j * qⱼ)
```

This allows embedding new documents without re-training the entire model.

---

## 5. Complexity

- Softmax: O(|V| * r)
- Negative sampling: O((|C| + |N|) * r), with small |C| and |N| ≈ O(r)

---

This method forms the basis of many document embedding approaches used in NLP. The original `w2v_p2vupdates.pdf` can be found in my [publications](/publications/).

