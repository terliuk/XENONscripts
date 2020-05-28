#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=12G
#SBATCH --output=/home/terliuk/logs/MClogs/log_WFsim_patterns.%A_%a.log
#SBATCH --account=pi-lgrandi
#SBATCH --job-name=WFsim_pattern
eval "$(/dali/lgrandi/strax/miniconda3/bin/conda shell.bash hook)"
conda activate strax
echo "STRAX environment activated"

if [ "$SLURM_ARRAY_TASK_ID" = "" ]; then
    echo "No SLURM ID, job is run interactively, assume 1"
    FILE_NR=0
    SLURM_ARRAY_TASK_ID=1
else
    FILE_NR=`expr $SLURM_ARRAY_TASK_ID - 1`
fi
FILE_NR=`printf "%06d\n" $FILE_NR`


if [ "$TMPDIR" = "" ]; then
    echo "TMPDIR is not set, setting to local folder"
    TMPDIR="."
fi

echo "Temporary directory : " $TMPDIR
echo "SLURM task ID : " $SLURM_ARRAY_TASK_ID

RUNID=$1
WFSIMCONFLINE=$2
NEVENTS=$3
echo "WFsim config: " $WFSIMCONFLINE

OUTFOLDER="/project2/lgrandi/terliuk/MCoutputs/WFsim_patterns/disk_meshes/${WFSIMCONFLINE}/"
OUTFILENAME="WFsim_${WFSIMCONFLINE}_N${NEVENTS}_${FILE_NR}.hdf5"
echo "Filename ID: " $FILE_NR
echo "OUTFOLDER   : " $OUTFOLDER 
echo "OUTFILENAME : " $OUTFILENAME

curdir=$PWD
echo "Current dir : " $curdir
echo "Changing to TMPDIR : " $TMPDIR
cd $TMPDIR
script=/home/terliuk/XENONscripts/WFsim_pattern_gen/generate_patterns.py
$script -c $WFSIMCONFLINE -o $OUTFILENAME -n $NEVENTS -f $SLURM_ARRAY_TASK_ID -r $RUNID

ls -l 
echo "Moving output file to final directory "
mv $OUTFILENAME $OUTFOLDER/$OUTFILENAME
echo "Removing strax data" 
du -h strax_data/
du -h resource_cache/
rm -r strax_data/
echo "Chainging back to : " $curdir
cd $curdir
conda deactivate
