# ion-cdft


## About the code

This repository contains code to train and perform classical density functional theory (cDFT), particularly hyperdensity functional theory for polar fluids.

## Citation

Please find the associated paper with the code:


***A. T. Bui, S. J. Cox, **"Dielectrocapillarity for exquisite control of fluids"**, Nat. Commun. **XX**, XXXXX (2026)***

Links to: [arXiv:2503.09855](
https://doi.org/10.48550/arXiv.2503.09855) | [Nat. Commun.](https://doi.org/XXXX)


## Contents
* `data`: Simulation data of density profiles for training.
* `training`: Code for training neural networks of the hyperdensity functionals.
* `models`: Keras models obtained from training.
* `cdft`: Performing cDFT calculation for structure and thermodynamics.


## Training data

Raw training data of the ML models are deposited on [Zenodo](https://zenodo.org/records/15085645).


## Installation

You can clone the repository with:
```sh
git clone https://github.com/annatbui/dielectrocapillarity-cdft.git
```

For training and evaluating the model, *TensorFlow/Keras* is used. Performance is better with a GPU.
To create a *conda* environment containing the required packages 

```sh
conda env create -f environment.yml
```


## License

This code is licensed under the GNU License - see the LICENSE file for details.




