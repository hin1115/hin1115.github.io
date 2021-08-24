---
title : '(Paper Review) Subgraph Nerual Network'
date: 2021-08-04
excerpt : ""
permalink: /posts/2021/08/Subgnn/
tags:
  - references
  - bash
use_math : True
---

# Reference
    1. Alsentzer, Emily, et al. "Subgraph neural networks." arXiv preprint arXiv:2006.10538 (2020).
    2. Hu, Weihua, et al. "Open graph benchmark: Datasets for machine learning on graphs." arXiv preprint arXiv:2005.00687 (2020).

# Introduction
## GNN Tasks

**GNNs** can be widely used for various task, Node classification, Link prediction, and Graph classification. Since these three tasks are fundamental and well-known methodology, Open Graph Benchmark(OGB) also separate published dataset into three types.

<p align="center">
  <img src="/assets/img/subgnn/subgraph_intro.png" alt="sub1" style="zoom:75%;" />
</p>


However, no datasets and architectures could cover **subgraph,** a subset of graph. Subgraphs have unique topology and properties, so it’s challenging to produce their representation. But still, subgraphs are important.

This paper proposed unprecedented concept of tasks, ***SUBGRAPH NEURAL NETWORK***, with novelty methodology about subgraph message passing.



### Problems

> *How to produce representation about subgraph?*

Graph level representation focuses on overall structure of graph, so it may loss the local structures.

Node level representation can characterize the local structure. But it’s unsure that it could capture the subgraph representation in big picture.

### Objectives

The main purposes of Subgraph Neural Network are

>*How we can generate powerful representations for subgraphs?*
>
>*How we can capture effectively the unique topology of subgraphs?*

## Subgraph

### What is Subgraph?

Graph $G$ contains subgraphs $\mathcal{S}=\\{S_1 \cdots S_5 \\}$.

Colors indicate subgraph labels $\mathcal{C}=\\{ C_1, C_2, C_3\\}$.

<p align="center">
  <img src="/assets/img/subgnn/subgraph1.png" alt="sub2" style="zoom:75%;" />
</p>


### How predict?

The task is predicting the subgraph properties by learning subgraph representations.

>  *Recognize and disentangle the heterogeneous properties of subgraphs*

>  *How they relate to underlying graph?*

### Topological standpoint of subgraph

1. Subgraphs require that we make **joint predictions** over larger structures of varying sizes.

   - *How to represent subgraphs that do not correspond with enclosing $k$-hop neighborhoods.*
   - *How can even comprise multiple disparate components.*

2. Subgraphs contain rich high-order connectivity patters, **both internally and externally.**

   - *How to inject information about **border and external*** into GNN's neural message passing.

3. Subgraphs can be **localized and reside** in one region of the graph or can be distributed across multiple local neighborhoods. 

   - *How to effectively learn about **subgraph positions** within underlying graph.*

4. Subgraph datasets include **dependencies** about subgraphs sharing edges and non-edges.
   - *How to incorporate these **dependencies** into the model architecture?*

## Background

### Basic of Graph

- Let $G=(V,E)$ denote a graph, both *undirected* and *directed*.
- $S = (V', E')$ is a subgraph of $G$ if $V'  \subseteq V $  and $E'  \subseteq E$.
- Each subgraph $S$ has a label $y_S$ and may consist of multiple connected components, $S^{(C)}$ which are defined as a set of nodes in $S$.

### Basic of GNN

- **Message Passing Network(MPN)** : MSG+AGG+UPDATE.
- **MSG** : node level messages to a node $v_i$ from neighborhood $N_{v_i}$. 
A message between node $(v_i, v_j)$ at layer $l$,  $m_{ij}^l = MSG(h_i^{l-1}, h_j^{l-1})$.
- **AGG** : messages from $N_{v_i}$ are aggregated and combined with $h_i^{l-1}$.
- **UPDATE** : produce $v_i$'s representation for layer $l$.



# SUBGNN

## Problem formulation

### Subgraph Embedding

- Specifies a neural message passing architecture to generate subgraph representation, $E_S : S$ → $\mathbb{R}^{d_s}$ 
- Uses the representations to learn a subgraph classifier $f : \mathcal{S} \xrightarrow{} \\{ 1,2, \cdots , C \\}$

### Properties of subgraph topology

Subgraphs have non-trivial ***internal structure***, ***border connectivity***, and notions of ***neighborhood*** and ***relative positions.*** 

### Six properties of subgraph topology
<p align="center">
  <img src="/assets/img/subgnn/subgraph2.png" alt="sub3" style="zoom:75%;" />
</p>


## Network

Property-specific messages $\text{MSG}_X^{A→S} = \gamma_X (S^{(C)},A_X ) \cdot \mathbf{a}_X$ are propagated from anchor patches $A_X$ to components of subgraph $S$.
<p align="center">
  <img src="/assets/img/subgnn/subgraph3.png" alt="sub1" style="zoom:75%;" />
</p>


### Key points

1. Novel aggregation scheme at the level of subgraph component.
2. *How to propagate neural message from a set of anchor patch to subgraph components?*
   → ***Entire subgraph representations could capture the six properties.***

### Subgraph-level message passing

- $A_{X} = \{ A_X^{(1)} , \cdots,A_X^{n_A} \} → A_{P}, A_{N}, A_{S}$.     
  **Anchor patches** are subgraphs, randomly sampled from $G$ in a channel-specific manner.

- Message from anchor patch $A_X$ to subgraph components $S^{(C)}$,
  $$
  \text{MSG}_X^{A \xrightarrow{}S} = \gamma_X (S^{(C)},A_X ) \cdot \mathbf{a}_X 
  $$

- These messages are transformed into order-invariant hidden representation $h_{X,c}$    
 
  $$
  g_{X,c}= \text{AGG}_M ( \{ \text{MSG}_X^{A_X  \xrightarrow{}S^{(C)}} \forall A_X \in \mathcal{A_X} \} )
  $$     
  
  $$
  h_{X,c} \xleftarrow{} \sigma(W_X \cdot [g_{X,c}; h_{X,c}])
  $$
  
- $h_{X,c}$ is a channel-specific hidden representation for components $S^{(C)}$ and channel $\text{X}$, and is passed to the next layer of the SUBGNN.

### Order invariace representation

- The order invariance of $\mathbf{h}_{X,c}$ is a necessary property for layer-to-layer message passing.
-  However, it limits the ability to capture the structure and position of subgraphs.
- ***Property-aware output representation***, $\mathbf{z}_{X,c}$ 
  - By producing a matrix of anchor-set messages $\mathbf{M}_{X}$ for each row.
  - Passing through a non-linear activation function.
  - Result : encode the structural or positional message from anchor patch $A$.
  - $z_{N,c} = h_{N,c} $ for the neighborhood channel.

### Aggregation

- Property-aware output representations $z_{X,c}$ are transformed as final subgraph representations.
  - $z_{X,c}$ $\xrightarrow{}$ Channel aggregation $\xrightarrow{}$ Layer aggregation $\xrightarrow{}$ READOUT $\xrightarrow{}$ $z_{S}$

## Point-Aware Routing

Each channel $X$(position, neighborhood, structure) has three key elements.

### 1. Sampling anchor patches

***Anchor patch sampling function $\phi_{X} : (G, S^{(c)} ) \xrightarrow{} A_X$***

#### Position channel

- Internal anchor patch $\phi_{P_I} \xrightarrow{} A_{P_I}$ : shared across all components in $S$
- Border position anchor$\phi_{P_B} \xrightarrow{} A_{P_B}$ : shared across all subgraphs.

#### Neighborhood channel

- $\phi_{N_I}$ samples nodes from the subgraph components $S^{(C)}$.
- $\phi_{N_B}$ samples nodes from the $k$-hop neighborhood of $S^{(C)}$.

#### Structure channel

- Structure anchor patch sampler $\phi_{S}$  is used for both internal and border structure, resulting $\mathcal{A_{S_I}}$ and $\mathcal{A_{S_B}}$ are shared across all $S$.
- $\phi_{S}$ returns a connected component sampled from the graph via triangular random walks.

### 2. Neural encoding of anchor patch

***Anchor patch encoder $\psi_{X} :  A_X\xrightarrow{} \mathbf{a}_X$***

#### Position and Neighborhood channel

- $\psi_{X}$ and $\psi_{P}$ : mapping the node embedding of anchor patch node. 
- $\psi_{S}$ : representations of structure anchor patches.

#### Structure channel

-  $w$ fixed-length triangular random walks
  $\xrightarrow{}$ sequence of traversed nodes $(u_{\pi_w (1)}, \cdots u_{\pi_w (n)} )$. $\xrightarrow{}$ fed into a bidirectional LSTM $\xrightarrow{}$ final representation $\mathbf{a}_S$

### 3. Estimating similarity of anchor patches

***Similarity function $\gamma_X : ( S^{(C)}, A_X )$***

The relative weighting of each anchor patch in building the subgraph component representation.

#### Position channel

- Function of the shortest path between $S^{(C)}$ and $A_P$ ; $\gamma_P(S^{(C)}, A_P) = 1 / (d_{sp} S^{(C)}, A_P)+1)$.


#### Neighborhood channel

- $\gamma_N(S^{(C)}, A_N)$ is defined similarly with position channel.

#### Structure channel

- Ordered degree sequence for the $S^{(C)}$ and $A_P$ using normalized dynamic time warping (DTW) measure. : $\gamma_S(S^{(C)}, A_S )= 1 / (\text{DTW}(d_{S^{(C)}}, d_{A_S})+1)$

### Message Passing Algorithm.
<p align="center">
  <img src="/assets/img/subgnn/subgraph33.jpg" alt="sub5" style="zoom:75%;" />
</p>

# Experiments

## Datasets

Four **Synthetic datasets** and four **Real-world datasets** for subgraph classification.

### Synthetic datasets

Challenge the abilities for each dataset.

- ***DENSITY*** : internal structure of subgraphs
- ***CUT RATIO*** : border structure
- ***CORENESS*** : the average core number of the subgraph $\xrightarrow{}$ test border structure and position
- ***COMPONENT*** : the number of subgraph components, test internal and external position

### Real-world datasets
<p align="center">
  <img src="/assets/img/subgnn/SUBGRAPH4.png" alt="sub1" style="zoom:75%;" />
</p>


## Experimental Result

- Evaluation metric : Micro-F1
- Adopt existing node embedding method, *GIN* & *GraphSAINT*. 

### Synthetic

For Synthetic datasets, SUBGNN is implemented with GIN node embedding.
<p align="center">
  <img src="/assets/img/subgnn/subgraph5.png" alt="sub1" style="zoom:75%;" />
</p>

### Real-word datasets

 For Real-world datasets, SUBGNN is implemented with both GIN and GraphSAINT node embeddings.
<p align="center">
  <img src="/assets/img/subgnn/subgraph6.png" alt="sub1" style="zoom:75%;" />
</p>
