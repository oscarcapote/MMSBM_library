# MMSBM_library

Program to find the membership factors and the probabilities of connections in a bipartite network adding metadata to the nodes. This program finds the most plausible parameters of the Mixed-Membership Stochastic Block Model (MMSBM), that fits a dataset of links between nodes. Also you can add users' metadata that can help (or not) to the link prediction.
# Language
- Python >= 3.5

# Requirements
- pandas
- numpy
- numba (optional but *highly reccommended*)

# The link prediction problem and how we solve it

The problem that solves this program is the link prediction in a bipartite complex network where you have two types of nodes that can be users and items, politicians and bills... and labeled links that represents ratings, votes, preferences or simple connections.

The model that we use is Mixed-Membership Stochastic Block Model that supposes that nodes belongs to a superposition of groups and the probability of connection depends only of the groups that they belong (read [this][dc131834]).

  [dc131834]: https://www.pnas.org/doi/abs/10.1073/pnas.1606316113 "Accurate and scalable social recommendation using mixed-membership stochastic block models"

![bipartite](images/bipartite.png)

To get these model parameters we use an Bayesian inference approach to get the most plausible parameters, using the Maximum a Posteriori (MAP) algorithm.

Besides the known links, we can use the node attributes or metadata to improve or not the predictions. This improvement depends on the correlation of the metadata and the links.

![multipartite](images/multipartite.png)

In this case we extend our problem to a link prediction problem in a multipartite complex network where we have to take into account the metadata bipartites networks. Each metadata bipartite network is described using a MMSBM (read [this][f918b40b]).

  [f918b40b]: https://journals.aps.org/prx/pdf/10.1103/PhysRevX.12.011010 "Node Metadata Can Produce Predictability Crossovers in Network Inference Problems"

To adapt the algorithm to any situation where metadata is totally, partially or no correlated, our approach incorporates an hyperparameter _lambda_ to each metadata that informs about the importance of the metadata when are used.
![lambdes](images/lambdes.png)

You can read and run the ``Tutorial.ipynb``, see the example at ``Example.ipynb`` and read the documentation at [https://oscarcapote.github.io/MMSBM_library/index.html](https://oscarcapote.github.io/MMSBM_library/index.html).
