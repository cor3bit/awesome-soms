# Awesome SOM for ML [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of resources for second-order 
stochastic optimization 
methods in machine learning.




## Table of Contents
- [Books and Lecture Notes](#books-and-lecture-notes)
- [Papers](#papers)
- [Implementation in JAX](#implementation-in-jax)




## Books and Lecture Notes

- [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) by Jorge Nocedal and Stephen J. Wright, 2006 :dollar:
- [Introduction to Optimization and Data Fitting](https://www2.compute.dtu.dk/pubdb/pubs/5938-full.html) by H. B. Nielsen and K. Madsen, 2010
- [Optimization for Machine Learning](https://arxiv.org/abs/1909.03550) by Elad Hazan, 2019
- [Topics in Machine Learning: Neural Net Training Dynamics (Winter 2022)](https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2022/) by Roger Grosse, University of Toronto, 2022





## Papers

### Overview

- [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838) 
by Léon Bottou, Frank E. Curtis, Jorge Nocedal, 2016.

- [Exact and inexact subsampled Newton methods for optimization](https://academic.oup.com/imajna/article/39/2/545/4959058) 
by Raghu Bollapragada, Richard H Byrd, Jorge Nocedal, 2018.



### Diagonal Scaling

- [AdaHessian: An Adaptive Second Order Optimizer for Machine Learning](https://arxiv.org/abs/2006.00719) 
by Zhewei Yao, Amir Gholami, Sheng Shen, Mustafa Mustafa, Kurt Keutzer, Michael W. Mahoney, 2020. Algorithm: AdaHessian

- [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342) 
by Hong Liu, Zhiyuan Li, David Hall, Percy Liang, Tengyu Ma, 2023. Algorithm: Sophia



### Hessian-free Optimization

- [Learning Recurrent Neural Networks with Hessian-Free Optimization](https://www.cs.toronto.edu/~jmartens/docs/RNN_HF.pdf) 
by James Martens, Ilya Sutskever, 2011.

- [Training Neural Networks with Stochastic Hessian-Free Optimization](https://arxiv.org/abs/1301.3641) 
by Ryan Kiros, 2013. Algorithm: SHF



### Quasi-Newton

- [A Stochastic Quasi-Newton Method for Large-Scale Optimization](https://arxiv.org/abs/1401.7020) 
by R.H. Byrd, S.L. Hansen, J. Nocedal, Y. Singer, 2014. 

- [A Multi-Batch L-BFGS Method for Machine Learning](https://arxiv.org/abs/1605.06049) 
by Albert S. Berahas, Jorge Nocedal, Martin Takáč, 2016. 

- [Practical Quasi-Newton Methods for Training Deep Neural Networks](https://arxiv.org/abs/2006.08877) 
by Donald Goldfarb, Yi Ren, Achraf Bahamou, 2020.



### Gauss-Newton

- [Efficient Subsampled Gauss-Newton and Natural Gradient Methods for Training Neural Networks](https://arxiv.org/abs/1906.02353) 
by Yi Ren and Donald Goldfarb, 2019. Algorithm: SWM-GN, SWM-NG

- [On the Promise of the Stochastic Generalized Gauss-Newton Method for Training DNNs](https://arxiv.org/abs/2006.02409) 
by Matilde Gargiani et al., 2020. Algorithm: SGN

- [Stochastic Gauss-Newton Algorithms for Nonconvex Compositional Optimization](https://arxiv.org/abs/2002.07290) 
by Quoc Tran-Dinh et al., 2020. Algorithm: SGN with SARAH estimators

- [Nonlinear Least Squares for Large-Scale Machine Learning using Stochastic Jacobian Estimates](https://arxiv.org/abs/2107.05598) 
by Johannes J. Brust, 2021. Discusses using stochastic Jacobian estimates in nonlinear least squares for scalable machine learning.
Algorithm: NLLS1, NLLSL

- [Improving Levenberg-Marquardt Algorithm for Neural Networks](https://arxiv.org/abs/2212.08769) 
by Omead Pooladzandi and Yiming Zhou, 2022. Algorithm: LM

- [Rethinking Gauss-Newton for learning over-parameterized models](https://arxiv.org/abs/2302.02904) 
by Michael Arbel et al., 2023.




### Fisher Information

- [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671) 
by James Martens and Roger Grosse, 2015. Algorithm: K-FAC

### Other

- [Second-order optimization with lazy Hessians](https://arxiv.org/abs/2212.00781) 
by Nikita Doikov, El Mahdi Chayti, Martin Jaggi, 2022.



## Implementation in JAX

- [Optax](https://github.com/google-deepmind/optax) - mostly first-order accelerated methods

- [JAXopt](https://github.com/google/jaxopt) - deterministic second-order methods (e.g., 
Gauss-Newton, Levenberg Marquardt), stochastic first-order methods PolyakSGD, ArmijoSGD

- [KFAC-JAX](https://github.com/google-deepmind/kfac-jax) - implementation of KFAC from the DeepMind team

- [AdaHessian](https://github.com/nestordemeure/AdaHessianJax) - implementation of the AdaHessian optimizer

