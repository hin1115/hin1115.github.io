---
title : '(Paper Review) Multitask Graph Autoencoder'
date: 2021-08-20
excerpt : ""
permalink: /posts/2021/08/mga/
tags:
  - references
  - bash
use_math : True
---


## Reference
    Chen, Shicong, et al. "Link Prediction and Node Classification Based on Multitask Graph Autoencoder." Wireless Communications and Mobile Computing 2021 (2021).

## About

> - In network representation learning, node classification can be leveraged weighted edges.
> - However, most of node classification data such as OGB dataset, do not include edge information.
> - So, we need to extract the edge information without supervised learning.
> - Is the unsupervised learning for link prediction possible?
> - And then, is it helpful for node classification?

## Link prediction

### classical similarity index

- CN : **common neighbor** to predict the potential links between node pairs.
- AA : **penalty** on lower-connected neighbors.
- Jaccard : **similarity** by comparing the proximities and differences between sample sets of common neighbors.
- LP : introduces the influencing factor of a **third-order local path** to the algorithm.
- Katz : improves the prediction accuracy by optimizing LP index, and extend the local path to **global path**
- RA : kinetic equilibrium state with power-law strength-degree correlation

By measuring the similarities between embedding vectors, We can find the potential correlations !



## Auto Encoder

### Learn representations

An auto encoder is implemented to reduce vector-dimension from $R^N$ to $R^D$.
In this procedure, there are three characteristics.
<p align="center">
  <img src="/assets/img/multitaskautoencoder/multi1.PNG" alt="multi1" style="zoom:75%;" />
</p>


1. The low-dimensional embedding space will be decoded as reconstruction vector $\hat{x_i}$.

2. Weights in encoder and decoder are shared among each nodes.(Parameter sharing)

3. Loss functions for structural deep network embedding.

   - First-order : **(Supervised)** capture local structure features by judging whether nodes are linked by a direct edge.

  $$
 Loss_{1st} = \sum_{i,j=1}^n s_{i,j}   || y_i^{(k)} - y_j^{(k)} || ^2_2 = 2 \text{tr}(Y^T LY)
  $$
  
  where $L$ is the Laplace vector matrix, $s_{i,j}$ is the element of the adjacency matrix, $Y$ is the encoded vector matrix of the hidden layer.
     
   - Second-order : **(Unsupervised)** preserves global structure features by observing the differences between the neighborhood structure of nodes.    
   
  $$
  Loss_{2nd} = \sum_{i=1}^n    || (\hat{x_i} - x_i)\odot b_i  || ^2_2 = || (\hat{X} - X) \odot B|| ^2_F
  $$
  
  where $\hat{x_i} - x_{i}$ is the reconstruction error, $\odot$ denotes the Hadamard product, and $b_i$ is a penalty coefficient.
     
 

## Link-prediction Algorithm

By utilizing similarity index and edge list, the final similarity matrix can be obtained.

1. $\alpha A^{3}$ : attenuation parameter with third-order path adjacency matrix.
2. $S_{MAARA}$ : Optimized common neighbor matrix from AA index and RA index.
   = node similarity
3. $S_{PA}$ : matrix from preferential attachment using the degrees of pairwise nodes.
   = the likelihood one link connecting pairwise node

Finally, the similarity matrix can be calculated as 
$$
S_{MNLP} = S_{MAARA} * S_{PA} + \alpha A^{3}
$$

## Node-classification Algorithm

### Adjacency matrix vs Similarity matrix

> The adjacency matrix only describes the actual condition. However, the similarity matrix reveals ***the hidden structural similarity*** of the network!

### Deficiency in second-order proximity

> Measuring similarities whether an edge exists between pairwise node is not enough!

Actually, there are more things to consider bringing about deviations.

- Properties of common neighbor
- Length of path
- Attenuation parameters.

### High-order proximity

High-order loss function seeks the potential link by utilizing similarity matrix from link prediction.    
$$
Loss_{high-order} = \sum_{i=1}^n || ( \hat{x_i}-x_i ) \odot M * \gamma ||_2^2
$$ 

where $\gamma$ is the adjustment parameter.

### Node classification

Finally, the multi-task graph autoencoder can classify each node labels using the combined loss function.
$$
Loss_{mix} = \alpha Loss_{1st} + Loss_{2nd} + Loss_{high-order}
$$
<p align="center">
  <img src="/assets/img/multitaskautoencoder/multi2.PNG" alt="multi2" style="zoom:75%;" />
</p>

