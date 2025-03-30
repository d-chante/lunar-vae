# lunar-vae

A Variational Autoencoder implementation for training on LRO Diviner temperature data. 

## Setup

These instructions assume a Ubuntu 24.04 LTS operating system.

1. Install git: `sudo apt update && sudo apt install git`
2. [Install Docker](sudo apt update && sudo apt install git)
3. Clone this repository: `git clone git@github.com:d-chante/lunar-vae.git`
4. Create an `outputs` folder: `mkdir outputs`
5. Go to the docker folder: `cd /path/to/lunar-vae/docker`
6. Edit `common.sh` file to reflect the locations of the relevant folders
7. Build the Docker container by running: `./builsh`
8. Start the Docker container by running `./start.sh`

## Usage

To generate `profile_list.json`: 
```
python3 scripts/create_profile_list.py -d /path/to/profile/jsons/dir -o /path/to/output/dir
```

To convert Profiles in `profile_list.json` to a single csv file:
```
python3 scripts/convert_json_to_csv.py -d /path/to/profile/jsons/dir -i /path/to/input/profile_list.json -o /path/to/output/dir
```

To train, first, make relevant changes to `config/cosmocanyon_cfg.yaml`, then:
```
python3 scripts/train_vae.py -c /path/to/config/cosmocanyon_cfg.yaml -s
```

Other examples can be found in [dubois-masc-f2025](https://github.com/d-chante/dubois-masc-f2025)

## References & Acknowledgements
- Implementation based on the paper [Unsupervised Learning for Thermophysical Analysis on the Lunar Surface (2020)](https://iopscience.iop.org/article/10.3847/PSJ/ab9a52/pdf)
- Credit to [arushisinha98](https://github.com/arushisinha98) whose [latent space sampling/visualization](https://github.com/arushisinha98/variational-autoencoder) I borrowed from, as well as a big thank you for your time and advice in troubleshooting