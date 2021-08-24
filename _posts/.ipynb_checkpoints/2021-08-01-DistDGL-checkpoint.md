---
title : '(Paper Review) DistDGL &#58; Distributed Graph Neural Network Training for Billion-Scale Graphs'
date: 2021-08-01
excerpt : ""
permalink: /posts/2021/08/disg_dgl/
tags:
  - references
  - bash
use_math : True
---

# Reference
    1. Zheng, Da, et al. "Distdgl: distributed graph neural network training for billion-scale graphs." 2020 IEEE/ACM 10th Workshop on Irregular Applications: Architectures and Algorithms (IA3). IEEE, 2020.
    2. Hu, Weihua, et al. "Open graph benchmark: Datasets for machine learning on graphs." arXiv preprint arXiv:2005.00687 (2020).
    3. Yook, D., Lee, H., & Yoo, I.-C. (2020). 심층 신경망 병렬 학습 방법 연구 동향. 한국음향학회지, 39(6), 505–514. https://doi.org/10.7776/ASK.2020.39.6.505

# Introduction
## GNN for Large Graphs
Graph Neural Networks(GNNs) have shown success in learning from graph-structured data,
and have been applied to many graph applications.
However, training a GNN model on a large graph is still challenging, because

- Graph has depending samples, so it's difficult to distribute as mini-batch.
- Read hundreds of neighbor vertex data to compute a single vertex representation.

*So, how to prune the vertex dependency while maintaining representation?*

## More problems

> How we can calculate SGD like typical neural network?
>
> Are the existing framework suitable for giant graph which are made for single machine?
>
> How we can avoid bottleneck problem?

## Mini-batch training

GNN mini-batch training has its own characteristics.

1. Sample a set of ***N*** vertices, *called target vertices*.
2. Randomly pick at most ***K*** neighbor vertices for each target vertex.
3. Compute the target vertex representations by gathering messages from the sampled neighbors.

Also, we have to consider *reclusive computation* in sampling.

<!-- ![](/assets/img/DistDGL/dgl1.png) -->
<p align="center">
  <img src="/assets/img/DistDGL/dgl1.png" alt="dg1" style="zoom:40%;" />
</p>
In this work, DistDGL is introduced on top of Deep Graph Library(DGL) to perform efficient and scalable mini-batch GNN training on a cluster of machines.

In summarize, here are so many contribution of DistDGL.

## DistDGL

- Scalable mini-batch GNN training.

- Multiple optimizations (to speed up computing).

- Distributed graph data.

- Synchronous training approach and *ego-network* forming .

- Multiple load balancing optimizations(to tackle the imbalance issue).

- Reduce network communication in sampling, using *halo node*.

- Distributed embedding  for transductive graph.

  

*So, how DistDGL works?*

# DistDGL System Design

## Distributed Training Architecture

Basically, DistDGL follows the *synchronous stochastic gradient descent(SGD)* training.
It means that each machine computes model gradients w.r.t its own mini-batch.

At a high level, DistDGL consists of the following logical components.

### Logical components

- ***Sampler*** : sampling the mini-batch graph structures
- ***KVStore*** : stores all vertex data and edge data, distributedly.
- ***Trainer*** : compute the gradients of the model parameters over a mini-batch
- ***Dense model update component*** : aggregate dense GNN parameters to perform synchronous SGD.

<!-- ![](/assets/img/DistDGL/dgl2.png) -->
<p align="center">
<img src="/assets/img/DistDGL/dgl2.png" alt="dgl2" style="zoom:50%;" />
</p>
The purpose of *logical components* is reducing the network traffic among machines
because graph computation is data intensive.

So DistDGL adopts the owner-compute rule.

### Owner-compute rule

The general principle is to dispatch computation to the data owner to reduce network communication.

1. Partitions the input graph with a light-weight min-cut graph partitioning algorithm.
2. Partitions the vertex/edge features and co-locates them with graph partition.
3. Sampler, KVStore, and trainer works on each machine to serve local partition data.

<!-- ![](/assets/img/DistDGL/dgl13.png) -->
<p align="center">
<img src="/assets/img/DistDGL/dgl3.png" alt="dgl3" style="zoom:75%;" />
</p>


## Graph Partitioning

### Goal of graph partitioning

- Graph partitioning is a preprocessing step before distributed training.

- Split the input graph to multiple partitions.

- Minimal number of edges across partition.

<!-- ![](/assets/img/DistDGL/dgl4.png) -->
<p align="center">
<img src="/assets/img/DistDGL/dgl4.png" alt="dgl4" align="right" style="zoom:65%;" />
</p>

### METIS

DistDGL adopts METIS to partition a graph.

- METIS assigns densely connected vertices to the same partition to reduce the number of edge cuts between partitions.
- Consist of *Core vertices* and *HALO vertices*

Also, DistDGL deploys multiple strategies to balance the partitions. 

- METIS balances the number of vertices in a graph.
  → Insufficient for synchronous mini-batch tarining.
- **Adopt multi-constraint partitioning(user-defined constraints)**

## Distributed Key-Value Store

Because of the partition, we need to read data from remote partitions and access on other machine.(*refer to remote process call* ***(RPC)****)*

- KVStore serever use shared memory(shares all data with the trainer process)
- Trainers can access most of the data directly.
- Support sparse embedding for training transductive models with learnable vertex embeddings.

## Distributed Sampler

> So, how the sampling algorithm conducted?

Instead of using baseline(DGL), DistDGL proposed more implementations.

1. Trainers issues sampling request (target vertices in the current mini-batch).
2. Graph partitioning algorithm → core vertex assignment → requests are dispatched.
3. Requests on sampler servers  → sampling operator → result back to the trainer
4. Trainer collects the result and stitch them → **generate a mini-batch**.

### Multiple Optimization

>How sample mini-batches work in parallel? 
>How avoid the cost of the RPC stack?

1. Sampling request → local sampler server.
2. Trainers overlap the sampling cost with mini-batch training.
3. Sampling workers access the graph structure(stored on the local sampler server, via shared memory).
4. Sampling workers also overlaps the RPCs with local sampling computation.

## Mini-batch Trainer

Mini-batch trainers run on each machine to jointly *estimate gradients* and *update parameters* of user's models. 

The key points are,

1. split the training set distributedly,
2. generate balanced workloads between trainers.

Because of the balanced partitions and synchronous SGD, the distributed training does not affect convergence rate or the model accuracy.

### Balanced partition

> How maintain the balance between each trainer?

Note that each trainer has the same number of training samples.

DistDGL uses a two-level strategy.

1. Split the training samples based on their IDs (relabeled during graph partitioning)
2. Assign the ID range to a machine

<!-- ![](/assets/img/DistDGL/dgl15.png) -->
<p align="center">
<img src="/assets/img/DistDGL/dgl5.png" alt="dgl5" style="zoom:75%;" />
</p>
### Parameter synchronization

In this paper, DistDGL use both SGD method, **synchronous SGD** and **asynchronous SGD.**

- Synchronous SGD : to update dense model parameters
- Asynchronous SGD : to update the sparse vertex embeddings in the Hogwild fashion to overlap communication and computation.

<!-- ![](/assets/img/DistDGL/dgl67.png) -->
<p align="center">
<img src="/assets/img/DistDGL/dgl67.png" alt="dgl67" style="zoom:75%;" />
</p>
# Experiments

Now, we evaluate DistDGL to answer the following questions.

> Can DistDGL train GNNs on large-scale graphs and accelerate the training with more machines?
>
> Can DistDGL's techniques effectively increase the data locality for GNN training?
>
> Can our load balancing strategies effectively balance the workloads in the cluster of machines?

## Dataset

The experiments are conducted on Open Graph Benchmark(OGB) Dataset. [[Link]](https://ogb.stanford.edu/)

OGB dataset includes several property prediction datasets about Node, Link, and Graph itself. 
In this paper, authors used two *Node property prediction* dataset and large one, ```product``` and ```papers100M```, respectively.
<!-- ![](/assets/img/DistDGL/dgl_data.png) -->
<p align="center">
<img src="/assets/img/DistDGL/dgl_data.png" alt="dgl_data" style="zoom:50%;" />
</p>
## Result

I'm gonna summarize by showing the result figures of the experimental result to show the performance of DistDGL.

**Training speed**

DistDGL vs Euler v2.0 on ```OGBN-PRODUCT``` graph.
<!-- ![](/assets/img/DistDGL/dgl555.png) -->
<p align="center">
<img src="/assets/img/DistDGL/dgl555.png" alt="dgl555" style="zoom:50%;" />
</p>
**Sparse embedding speed**

DistDGL vs Pytorch-Sparse on ```OGBN-PRODUCT``` graph with GRAPHSAGE.
<!-- ![](/assets/img/DistDGL/dgl10.png) -->
<p align="center">
<img src="/assets/img/DistDGL/dgl10.png" alt="dgl10" style="zoom:50%;" />
</p>
**Scalability**

DistDGL achieves linear speedup w.r.t the number of machines.
<!-- ![](/assets/img/DistDGL/dgl11.png) -->
<p align="center">
<img src="/assets/img/DistDGL/dgl11.png" alt="dgl11" style="zoom:50%;" />
</p>
**Training accuracy**

DistDGL quickly converges to almost the same peak accuracy achieved by the single-machine training.
<!-- ![](/assets/img/DistDGL/dgl12.png) -->
<p align="center">
<img src="/assets/img/DistDGL/dgl12.png" alt="dgl12" style="zoom:50%;" />
</p>
**METIS effectiveness**

METIS partitioning with multi-constraints to balance the partitions achieves good performance on both datasets.
<!-- ![](/assets/img/DistDGL/dgl13.png) -->
<p align="center">
<img src="/assets/img/DistDGL/dgl13.png" alt="dgl13" style="zoom:50%;" />
</p>


# Implementation

DistDGL doesn't support independent github. Instead, they embed their module into Deep Graph Library[(DGL)](https://docs.dgl.ai/en/0.6.x/index.html), with ```distributed``` section. 

Following the tutorials and documentation would be helpful. 

For me, ```DGL``` is much better than ```Pytorch-geometric```, which is very sensitive to conda environment(*e.g. Pytorch, Cuda*).
