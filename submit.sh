#!/bin/bash
#SBATCH --job-name=ScraperElena
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH --output=congreso_scraper_%j.log

# Activate the base conda environment
source /var/scratch/eei440/Anaconda3/etc/profile.d/conda.sh
conda activate base

# Move to your repo folder
cd /var/scratch/eei440/Anaconda3/PoliBiasES-BachelorProject-EIEN/

# Run the scraper script
python scraper.py

