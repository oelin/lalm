# LaLM

An implementation of LaLM (Latent Language Model) in PyTorch.

## Model

LaLM is a transformer-based variational autoencoder that leverages autoregression to learn hierarchical latent codes. According to the [chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability)), the description length of any string can be decomposed into a sum of conditional description lengths:

$$L(x_1, \dots, x_n) = L(x_1) + L(x_2|x_1) + \dots + L(x_n|x_1, \dots, x_{n-1}).$$

This formulation elucidates the possibility of generalized residual vector quantization, where an autoregressive model generates a series of latent codes, each depending on *all* codes preceeding it. Whereas conventional RVQs are Markovian and use a *fixed* transformation between consecutive quantizers, this scheme allows for learned transformations, parameterized by deep neural network.
