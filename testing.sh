#!/bin/bash
#SBATCH --job-name=ScraperElena
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH --output=congreso_scraper_%j.log

# Load environment
module load Python/3.13.2  # Adjust to your cluster's Python version
conda activate      

# Go to your GitHub repo directory
cd /var/scratch/eei440/Anaconda3/PoliBiasES-BachelorProject-EIEN/

# Make sure your CSV is in place
cp /var/scratch/eei440/Anaconda3/PoliBiasES-BachelorProject-EIEN/clean_ALL_complete_congreso_links_motion_titel_category_subcat_motionid.csv .

# Run the scraper
python scraper.py