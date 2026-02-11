# Locat #
This package is an implementation of Locat. Locat identifies marker genes with
concentrated expression domains while excluding broadly expressed background
genes across diverse biological contexts.


## Installation ## 
The main version of the package can be installed as
```
pip install locat
```

The package is based on jax (see [installation instructions](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html#installation)). 
To enable jax with GPU or TPU, install jax with cuda support.  
For example, if your system has `cuda12`, install locat as:
```
pip install locat jax[cuda12]
```

The development version of the package can be installed as
```
pip install git+https://github.com/Klugerlab/Locat.git
```
