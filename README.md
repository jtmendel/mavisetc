# MAVISETC
This provides exposure time estimates and data simulation capabilities for MAVIS, a new optical imager and IFS being developed for the ESO VLT.

Current version: v1.0.1a0

# Installation
In order to install all components of the ETC you will need both git and git-lfs installed. With those running you can install an editable version of the software like so:
```
git clone https://github.com/jtmendel/mavisetc.git
cd mavisetc
git lfs checkout
python -m pip install -e .
```
This will compile the necessary routines and create a symlink to their location in your python directory. The ```git lfs checkout``` call will fetch the bundled stellar library and etalon models (~530 mb total). If you don't plan to use them then this should not be needed (but note that one of the examples in the included notebook will not work!). If you chose, you can also perform standard (non-editable) installation, i.e:
```
python -m pip install .
```
As well as Numpy and Scipy, you will need to have the following Python modules installed
* [astropy](http://www.astropy.org/) - Python tools for astronomy.
* [skycalc_cli](https://www.eso.org/observing/etc/doc/skycalc/helpskycalccli.html) - The command line interface to ESO's advance sky model.

# Contents
* Description coming soon, I promise!
* `doc`: contains (eventually) a manual, installation instructions, and version history, blah blah blah.
