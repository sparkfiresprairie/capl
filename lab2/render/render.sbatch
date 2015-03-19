#!/bin/bash
#SBATCH -J render           # job name
#SBATCH -o render.output       # output and error file name (%j expands to jobID)
#SBATCH -n 1              # total number of mpi tasks requested
#SBATCH -p gpudev     # queue (partition) -- normal, development, etc.
#SBATCH -t 01:00:00        # run time (hh:mm:ss) - 1 hours
#SBATCH --mail-user=someemailaddrofyours@gmail.com
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH -A EE382N-20
./checker.pl 
