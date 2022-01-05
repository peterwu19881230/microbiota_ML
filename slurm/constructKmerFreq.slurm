#!/bin/bash                                                                                                           
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION                                                                           
#SBATCH --export=NONE        #Do not propagate environment                                                            
#SBATCH --get-user-env=L     #Replicate login environment                                                             

##NECESSARY JOB SPECIFICATIONS                                                                                        
#SBATCH --job-name=JobExample1     #Set the job name to "JobExample1"                                                 
#SBATCH --time=12:00:00                                                                                               
#SBATCH --ntasks=1                 #Request 1 task                                                                    
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node                                                     
#SBATCH --mem=64G              #per node                                                                              
#SBATCH --output=kmer_NB.%j    #Send stdout/err to "Example1Out.[jobID]"                                              

#First Executable Line                                                                                                
module load Anaconda3/2021.05
eval "$(conda shell.bash hook)" # https://hpc-unibe-ch.github.io/software/Anaconda.html                               


#timer start                                                                                                          
SECONDS=0
python grace_compute_kmer_frequencies.py

echo "===== (Job done. Run time: $SECONDS seconds)  ====="
