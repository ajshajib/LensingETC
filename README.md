# Lensing Exposure Time Calculator

The `LensingETC` tool allows to optimize observing strategies for multi-filter 
imaging campaigns of galaxy-scale strong lensing systems. See Shajib et al.
(2022) for a description of the tool. See the notebook [Lensing Exposure 
Time Calculator.ipynb](https://github.com/ajshajib/LensingETC/blob/main/Lensing%20exposure%20time%20calculator.ipynb) for an example of how to setup and run the tool.

`LensingETC` is built on top of `lenstronomy`([GitHub repo](https://github.com/sibirrer/lenstronomy)), 
a lens modelling software program affiliated 
with Astropy. See [here](https://github.com/sibirrer/lenstronomy/blob/main/AUTHORS.rst) 
for a full list of contributors to `lenstronomy`.


# Installation

This package does not need an installation, as it is intended to be used 
through a Jupyter notebook that is placed at the same directory of the file 
`lensing_etc.py`. However, `lenstronomy` needs to be installed, which can be 
done with:

```
pip install lenstronomy
```

You can also check the 
`lenstronomy`([GitHub repo](https://github.com/sibirrer/lenstronomy)) page if you
want to install a bleeding-edge version.

# Citation

If you have used `LensingETC` in planning an observing campaign, we request 
you to cite the following paper in your publication that uses data from the observing campaign:

- [Shajib et al. (2022)](https://arxiv.org/abs/2203.05170)
