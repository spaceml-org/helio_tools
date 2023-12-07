# HelioTools (In Progress)
[![CodeFactor](https://www.codefactor.io/repository/github/jejjohnson/helio_tools/badge)](https://www.codefactor.io/repository/github/jejjohnson/helio_tools)
[![codecov](https://codecov.io/gh/jejjohnson/helio_tools/branch/main/graph/badge.svg?token=YGPQQEAK91)](https://codecov.io/gh/jejjohnson/helio_tools)

> This package has some simple, minimal preprocessing of helio-data to make it machine learning ready.


---
## Demos

> Some demos showcasing how we can download some data, do some preprocessing regimes and integrate with ML datasets.

#### Data Downloading

**Downloading Data**. 
In this notebook, we have a demonstration of how we can download SDO data.
See [notebook](./notebooks/0.1_data_download_sdo.ipynb) for details.

#### Data Preprocessing

**Preprocessing Data**. 
In this notebook, we have a demonstration of how we can preprocess the data using a sequence of transformations.
We showcase a series of tested transformations which have had success for ML applications (e.g. [ITI](https://github.com/RobertJaro/InstrumentToInstrument/tree/master))
See [notebook](./notebooks/1.1_preprocessing_sdo.ipynb) for details.

**Preprocessing Configurations**.
In this notebook, we demonstrate how we can create configurations for these transformations.
In particular, we demonstrate how `Hydra-Zen` can be used to help facilitate readable transformations. 
See [notebook](./notebooks/2.1_preprocess_configs.ipynb) for details.

#### Machine Learning

**Numpy DataLoader**.
We demonstrate how we can use data to create a simple dataloader using numpy files downloaded.
See [notebook](./notebooks/3.1_numpy_dsdl.ipynb) for details.

**RasterVision**.
We demonstrate how we can use a more complex and advanced dataloader regime from `rastervision`.
In particular, we showcase how we can create independently sampled time series images in addition to time series images.
See [notebook](./notebooks/3.2_rastervision.ipynb) for details.

---
## Installation

We can install it directly through pip

```bash
pip install git+https://github.com/spaceml-org/helio_tools
pip install gsutil
```

We also use poetry for the development environment.

```bash
git clone https://github.com/spaceml-org/helio_tools
cd helio_tools
conda create -n helio_tools python=3.11 poetry
conda activate helio_tools
poetry install
```

---
## Test Dataset
We provide a test dataset for the notebooks containing data from SDO/AIA, EUI/FSI, EUI/HRI and PROBA2/SWAP which can be downloaded with gsutil 
```bash
gsutil cp -r gs://iti-dataset/ [local_path]

---
## References

**Software**

* [InstrumentToInstrument](https://github.com/RobertJaro/InstrumentToInstrument/tree/master) - Instrument-to-Instrument Translation.

**Glossary**

* [SDO](https://sdo.gsfc.nasa.gov/) - Solar Dynamics Observatory.
* [AIA](https://sdo.gsfc.nasa.gov/data/) - Atmospheric Imaging Assembly.
* [HMI](https://sdo.gsfc.nasa.gov/data/) - Helioseismic and Magnetic Imager.
* [EVE](https://lasp.colorado.edu/home/eve/data/) - Extreme Ultraviolet Variability Experiment.
* [SolO](https://sci.esa.int/web/solar-orbiter) - Solar Orbiter.
* FSI - Full Sun Imager.
* [SOHO](https://soho.nascom.nasa.gov//) - Solar and Heliospheric Observatory.
