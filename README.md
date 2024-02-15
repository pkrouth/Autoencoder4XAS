# AutoEncoder for XAS

X-ray absorption spectroscopy (XAS) is a premier materials characterization technique to study local structure of atomic configuration in nanomateials and bulk. While the extended portion of XAS is more common for quantitative analysis near edge portion (XANES) neural networks are recently applied. This package utilizes Autoencoder based unsupervised learning to encode XAS data in low-dimensional latent space and enable interpretation and downstream task such as extract of structural descriptors.<break>

![image](https://github.com/pkrouth/Autoencoder4XAS/assets/20447207/c6cb7cb0-80f5-41d7-83c8-e7ff4860c55d)


For more details on this ML application on XAS, please refer to the following paper:

- [Routh, Prahlad K., Yang Liu, Nicholas Marcella, Boris Kozinsky, and Anatoly I. Frenkel. "Latent representation learning for structural characterization of catalysts." The Journal of Physical Chemistry Letters 12, no. 8 (2021): 2086-2094.](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.0c03792)


---

Strucutre of xas_encoder

src
- models
   - Models defined separately
- train
    - Custom training file
- utils
    - Helper functions
- xasdata
    - Dataloaders
    

[//]: <> ( What should be a good and intuitive way to organize it? Like a tutorial? Or like a package? )


