# DeepLEAF-B: 
## ***Deep*** ***L***earning ***E***nabled ***A***nalysis ***F***ramework for ***B***-Meson Decays

This project aims to develop a deep learning (Neural Network) based analysis framework for B-Meson decays.

## Description

In place of physics based simulations and template fits we present the use of a Conditional Variational Auto-Encoder to generate a 4-Dimensional Histogram of true observables and a dual annealing algorithm for regression using this network. A Graph Neural Network extends from True Observable so detector-smeared and measured observables. Both networks' hyperparameters are Bayesian-Optimised using python package Optuna.

Current build status is a proof of concept for cSR and cVR for the _b&rarr;cμ<sup>-</sup>ν<sub>μ</sub>_ and _b&rarr;cτ<sup>-</sup>ν<sub>τ</sub>_ decay modes. 

## Getting Started

### Dependencies

* Python Version: 3.9.20

* PyTorch Version: 2.5.0
* Pandas Version: 2.2.2
* Optuna Version: 4.2.0
* Matplotlib Version: 3.9.2
* NumPy Version: 1.26.4

### Installing

Direct build processes are not yet fully developed. 

Clone repository using:
```
git clone https://github.com/muk03/DeepLEAF-B.git
```

### Executing program


```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
