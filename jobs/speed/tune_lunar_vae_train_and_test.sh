#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <tuning id>"
    exit 1
fi

tuning_id="$1"
config_dir="/home/d/d_chante/d_chante/lunar-vae/config/tuning/${tuning_id}"

if [ ! -d "$config_dir" ]; then
    echo "Error: Directory $config_dir does not exist."
    exit 1
fi

config_files=($(ls ${config_dir}/*.yaml))

for config_file in "${config_files[@]}"; do
    # Add a slight delay so that each job
    # has a unique label (HH:mm:ss)
    sleep 10
    sbatch <<EOF
#!/encs/bin/tcsh

#SBATCH --job-name=lvae_train_and_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chantelle.dubois@mail.concordia.ca
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gpus=1

# Load anaconda module
module load anaconda3/2023.03/default

# Logging env
env

# Activate conda env
conda activate /home/d/d_chante/d_chante/env/lvae

# Run job
srun python /home/d/d_chante/d_chante/lunar-vae/scripts/train_and_test.py -c $config_file

# Deactivate the environment
conda deactivate

# End job
exit
EOF
done