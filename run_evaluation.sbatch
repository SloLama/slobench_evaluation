#!/bin/bash
#SBATCH --job-name=slobench_evaluation
#SBATCH --account=st2311-ponj-users
#SBATCH --output=evaluation_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=56G
#SBATCH --partition=gpu


container_path=/ceph/hpc/data/st2311-ponj-users/containers
nemo_version=23.10
container=${container_path}/slopt_nemo_${nemo_version}.sif

models_dir=/ceph/hpc/data/st2311-ponj-users/models

experiment_config=$1

output_file=$2

srun \
  --cpu-bind=verbose \
  --output="logs/${experiment_config}_eval.txt" \
  singularity exec \
    -B ${models_dir}:/models \
    $container \
    python src/evaluation_engine.py --config=$experiment_config --output_file=$output_file