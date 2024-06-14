#!/encs/bin/tcsh

#SBATCH --job-name=setup_env
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chantelle.dubois@mail.concordia.ca
#SBATCH --chdir=/speed-scratch/d_chante/tmp
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

# Load anaconda module
module load anaconda3/2023.03/default

# Init conda shell
conda init tcsh

# Re-source shell
source ~/.tcshrc

# Create conda env
conda create --prefix /speed-scratch/d_chante/env/lvae --yes

# Activate conda env
conda activate /speed-scratch/d_chante/env/lvae

# Install pip
conda install pip

# Install pcakages from requirements.txt
pip install -r /speed-scratch/d_chante/lunar-vae/support/requirements.txt

# Deactivate the environment
conda deactivate

# End job
exit