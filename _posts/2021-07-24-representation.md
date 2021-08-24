---
title : 'What is represenation learning in deep learning?'
date: 2021-07-24
excerpt : ""
permalink: /posts/2021/07/representation_learning/
tags:
  - representation learning
  - machine learning
use_math : True
---

# Reference
    Representation learning : A review and new perspective, Bengio 2014
    Learning Deep Architectures for AI, Bengio 2009
    
    
[Quora](https://arxiv.org/abs/2105.03300) by Ajit Rajasekharan.    
This post is a re-creation of the *Quora's* post above.  
[Paper Link(2014)](https://arxiv.org/pdf/1206.5538.pdf)     
[Paper Link(2009)](https://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf)

# Question 

> "What is representation learning in deep learning?"
>
> That is, deep learning is one of the many ways to learn representation.
>
> And "depth" in deep learning just happens to be one of the many factors to learning a good representation, even though it is an important one.

# What is representation learning?

Representation learning is learning representations of input data typically by transforming it or extracting features from it, that makes it easier to perform a task like classification or prediction. There are various ways of learning different representations.

For instance,

- in the case of probabilistic models, the goal is to learn a representation that captures the probability distribution of the underlying explanatory features for the observed input. Such a learned representation can then be used for prediction.
- in deep learning, the representations are formed by composition of multiple non-linear transformations of the input data with the goal of yielding abstract and useful representations for tasks like classification, prediction etc.

Focusing specifically on deep learning, representation learning is the consequence of the function a model learns where the learning is captured in the parameters of the model, as the function transforms input to output during training. Representation learning here is referring to the nature/characteristics of the transformed input - not the model parameters/function that is causal to it. The causal role is played both by the architecture of the model, and the learned parameters in mapping input to output. 

# Why is it important?

- The performance of any machine learning model is critically dependent on the representations it learns to output. The representation it learns to output in turn is directly dependent on the model and on what it is fed as input which could be raw data or the output o an upstream model(e.g. transfer learning) that transforms the raw input data.
- For example, in many deep learning models, one may wonder why is a simple linear layer stacked on top of a complex block with many non-linear layers of different kinds. It is typically because the complex block transforms the input to a rich representation which then just requires a simple linear layer to do task specific separation. Without the transformation performed by the complex block, it would not be possible to extract the key abstract features to just linearly separate them. A concrete example of this is the current state-of-the-art model, BERT for NLP tasks. The model outputs rich representations of input that can be used for a variety of NLP tasks with fine-tuning using very little task specific data and hardly any task specific architecture. 
- Understanding the different kinds of representations including what makes a particular representation good for a specific task, helps practitioners benefit from a broad cohesive view of the various deep learning model architectures.

# What makes a representation good?

The following priors (factors that are desired/assumed to be present) play a key role in outputting a good representation by learning a function $f$ that maps input $x$ to output representation $y$. Models may implement one or more of these priors to learn to output representations suited to a specific task.

+ **Smoothness** - This is perhaps one of the most basic priors present in machine learning where it is assumed that the learned function $f$ is smooth - meaning small changes in x leads to small changes in $f(x)$. Or equivalently $x≈y$ implies $f(x)≈f(y)$.
  + Machine learning algorithms that rely on the smoothness of the target function to be learned, require training examples to map out all the wrinkles in the target function. Generalization is achieved by local interpolation between neighboring training samples *(often called local estimators for this reason).* 
  + The smoothness assumption does not work when the target function to be learned is a highly varying function. That is, it has a lot of wrinkles *(ups and downs)* which may grow exponentially with the number of interacting features - this is often the case when the data is represented in the raw input space. This then requires a large number of training samples to capture all the wrinkles*(often called the curse of dimensionality)*.
  + However, models that make the smoothness assumption can be layered on top of models using more generic priors discussed below.
  + Models that make the smoothness assumption associate regions of the input space with their own private set of parameters *(e.g. clustering algorithms, decision trees etc.)*. The learned features are mutually exclusive. These models are for this reason learners of one-hot representations *(e.g. clustering algorithms output a one-hot representation of the input identifying which one of a small number of centroids best represents the input)*.
  + In summary, models *(local estimators)* that make smoothness assumption are one-hot representation learners requiring $O(N)$ parameters (and/or $O(N)$ examples) to distinguish $O(N)$ input regions.
  + One-hot representation is often associated with how the input is represented. Here we are examining models that learn one-hot representations.    
<p align="center"><img src="/assets/img/represenation_bangio.png"></p>

 + **Multiple explanatory features** - the input data distribution is generated by combination of several underlying features and a model that learns to compactly represent the input by a combination of these features, could potentially generalize without requiring as many examples as there are variations in the underlying function $f$. This compactness can only be achieved if the features are reused *(achieved by parameters being shared)* across examples that are not necessarily in local neighborhood *(this is unlike the local estimators with the smoothness assumption)*.
   + Models that learn distributed representations can achieve this compactness. For instance, deep learning models *(we will examine the depth prior below)* learn distributed representations of size $O(N)$ to distinguish $O(2^k )$ input regions where $k=N$  in a densely ditributed representation and $k<N$ in a sparsely distributed representation *(the boundary condition is a one hot representation where just one region is represented)*. In a distributed representation, a concept is represented by k features being turned on and an every feature participated in representing many concepts. Sparse representations are distributed representations where only $k<N$ features can be changed at any time. While both a dense distribuuted and one-hot representations allow for compositionality - the key benefit of distributed representations, a one-hot representation is one where all features are fully disentangled from each other - orthogonal to each other.
   + Clustering algorithms could also output distributed representations in the case of multiclustering where several clusterings are performed in parallel - an input belongs to multiple clusters. Another instance of generalization of clustering to distributed representation is if clustering is done across multiple regions of the input such as object recognition using a histogram of clustered categories detected in different regions of the input.
   + A concrete example of a "non-local" learner is the ***word2vec*** model - which is not even a deep model. It learns to represent words with distributed representations where words that are never in the local vicinity of each other in the training data, are mapped close to each other in the outputu space. These words *(that were never in the vicinity of each other in the input space)* but are close to each other in the output space tend to be words that are semantically similar, making this model very useful.
   + The price for this compactness is the difficulty in disentangling them. The compactness of distributed representations in part comes from capturing invariant features - which by definition have reduced sensitivity in the direction of invariance. Disentangling features however requires avoidance of information loss. So the approach to learning representations is a balance between two objectives - to disentangle as many features as possible sacrificing as little data as possible.
+ **Depth - a hierarchical organization of explanatory features** - We describe the world using a hierarchy of concepts, with more abstract concepts layered on top of less abstract ones.
   + Similarly, deep learning models learn functions that transform input to output using a composition of non-linear functions stacked in layers, where the output of layers form hierarchy of distributed representations with increasing levels of abstraction as input flows through them.
   + In addition to outputting progressive levels of abstract feature representations, deep learning architectures also enable feature reuse. Just as features are reused to represent different input regions in distributed representation, depth allows for feature reuse across layers by the multiple circuit paths in the computational graph from input to output through the nodes in the layers of the network.
+ **Semi-supervised learning.** Representations that are useful when learning $P(X)$ in an unsupervised manner tend to be useful when learning representations for $P(Y/X)$. Recent examples of this are models trained on an unsupervised language modeling task being then used to represent input for supervised tasks like classification, sequence tagging etc.
+ **Shared features across tasks.** Sharing of learned representation across task $P(Y\|X,task)$. Multi-task learning is an example.
+ **Manifold representations.** Even though input data for AI tasks such as image, text, audio reside in a high dimensional space *(e.g. 28x28 black and white image has 784 degrees of freedom yielding* $2^{784}$ *possible images)*, most uniformly sampled output *(for instance sampling from those* $2^{784}$ *possible images)* would not be naturally occuring images. The basic idea of manifold hypothesis is that there exists a lower dimensional manifold in which these naturally occuring images actually lie. So the  model learning task becomes learning to output representations that map the naturally occuring images in the high dimensional input space to the low dimensional manifold. The idea is that the small variations of the naturally occurring images (e.g. rotations etc) are mapping to corresponding changes in the learned representation in the low dimensional manifold. PCA is an example of a manifold mapping algorithm where the manifold is linear. Autoencoders are inspired by the manifold hypothesis and learn lower dimensional representations of high dimensional data. Even though autoencoders are known to perform dimensionality reduction, the manifold view gives a deeper understanding of this mapping.
+ **Natural clustering**. Different values of categorical variables such as object classes (e.g. cats, dogs) tend to be associated with separate manifolds. Each manifold is composed of learned representation of an object class (say dog, cat). So moving along a manifold tends to preserve the value of a category (e.g. variations of dog when moving oh the "dog" manifold). Interpolating across object classes would require going through a low density region separating the manifolds. In essence, manifolds representing object classes tend not to overlap much. This factor is exploited in machine learning. 
+ **Temporal and spatial coherence.** Identifying slowly moving/changing features in temporal/spatial data could be used as a means to learn useful representations. Even though different features change at different spatial and temporal scales, the values of the categorical variables of interest tend to change slow. So this prior can be used as a mechanism to force the representations to change slowly, penalizing change in values of categorical variables over time or space. Temporal coherence has used to model video.
+ **Sparsity**. For a given ovservation $x$, only a small set of possible features are relevant. This could captured in the representation by features that are often zero or by the fact the extracted features are insensitive to variations of $x$. Sparse autoencoders use this prior in the form of a regularization of the representation.
+ **Simplicity of Factor Dependencies.** If a representation is abstract enough, the features may relate to each other through simple linear dependencies. This can be seen in many laws of physics and this is the prior that is assumed when stacking a simple linear predictor on top of a learned representation that is rich and abstract enough. We see many instances of this in deep learning models - a simple linear layer slapped at the end of a deep convolutional model.

# What is a good criteria for learning representations?

From a practitioners perspective, one can have at least a qualitative sense if not a quantitative one of how good a model’s representations are by the different tasks it can be used for. For instance,

- the distributed representation of words (word embeddings) output by a simple model like word2vec with just two matrices as it basic architecture, has shown the power of non-local representation learning. Word embeddings have become the de facto representation of words for downstream NLP tasks
- The recent [BERT](https://arxiv.org/abs/1810.04805) model mentioned earlier is another example of how rich distributed representations output by the model can be used for a variety of NLP tasks with very little task specific data and hardly any task specific architecture, to obtain state-of-art results on NLP tasks.
- While BERT’s use case is one of reusing representations learnt from reconstruction of X from a masked version of it, in $P(Y/X,task)$, [GPT-2](https://openai.com/blog/better-language-models/) is another recent model that learns representations from a language modeling task and these representations are reused for tasks without any labeled data for tasks that are typically supervised. This is done by cleverly crafting the supervised task as a language modeling task *(predicting the next word given current sentence)*. While the performance of the model is not state-of-art yet on these tasks, it underscores the power and versatility of learnt representations, particularly distributed representations, which is largely responsible for the current successes in NLP.

