# APC Propeller Data

This repository contains data for APC propellers as well as a script to generate JSBsim propeller models from the provided APC propeller data. Thank you to APC for providing this data. The souce of this data can be found at [APC's website](https://www.apcprop.com/technical-information/file-downloads/).

## Prerequisites

- Create a python virtual environment

```bash
python3 -m venv .venv
```

- Activate the virtual environment

```bash
source .venv/bin/activate
```

- Install the required packages

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 read_apc_data.py --help
usage: read_apc_data.py [-h] propeller_path

Plot averaged Ct vs. J and generate JSB propeller XML model.

positional arguments:
  propeller_path  The path to the propeller data file or directory
                  containing multiple propeller data files.

options:
  -h, --help      show this help message and exit
```

run on a single propeller data file:

```bash
python3 read_apc_data.py perf_data/PERFILES2/PER3_5x4E-4.dat
```

run on a directory containing multiple propeller data files:

```bash
python3 read_apc_data.py perf_data/PERFILES2/
```

## Output

The script will generate a plot of the averaged Ct vs. J for the propeller data and a JSBsim propeller XML model file. In theory Ct vs J and Cp vs J should be the same. The experimental data shows fairly good agreement between RPM runs, however the is slight divergences. To account for this the JSBsim propeller model is generated using the average of the runs. the plots of the different runs are also generated for comparison.