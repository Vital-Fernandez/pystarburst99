(installation-doc)=
# Installation

pyStarburst99 can be installed by cloning the [GitHub repository](https://github.com/CalumHawcroft/Starburst):

```bash
git clone https://github.com/CalumHawcroft/Starburst.git
cd Starburst
pip install .
```

To also install the optional dependencies:

```bash
pip install ".[full]"
```

You can also run pyStarburst99 in your browser, with no installation, using this 
[Colab notebook](https://colab.research.google.com/drive/1MN4P8Q47jshh3bw2sEODb3eJDNxgAbLc?usp=sharing). The runtime 
is longer than with a local installation.

## Core dependencies

- [Numpy](https://numpy.org/install/) – Array operations
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html) – Reading the spectral libraries
- [Scipy](https://scipy.org/beginner-install/) – Interpolation and fitting
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html) – Plotting the results

## Optional dependencies

- [tqdm](https://tqdm.github.io/) – Progress bars in the terminal and notebooks (a plain text bar is used without it)

## Updating pyStarburst99

To upgrade to the latest version, pull the latest changes and reinstall:

```bash
git pull
pip install .
```

## Uninstalling

```bash
pip uninstall pystarburst99
```

## For developers

The following command installs the dependencies to compile the documentation and run the tests:

```bash
pip install ".[docs,tests]"
```

pyStarburst99 has been tested with **Python 3.12**. See the `pyproject.toml` in the repository for the most up-to-date 
build and dependency information.