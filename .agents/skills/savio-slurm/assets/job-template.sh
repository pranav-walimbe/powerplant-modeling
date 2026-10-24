#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=__PARTITION__
#SBATCH --qos=__QOS__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=__CPU_COUNT__
#SBATCH --time=__TIME_LIMIT__
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<EMAIL>
#SBATCH --output=/global/home/users/<USER>/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/<USER>/no2-modeling/logs/%x-%j.err

set -euo pipefail

cd "__REPOSITORY_PATH__"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate

srun python -u -m __PYTHON_MODULE__
