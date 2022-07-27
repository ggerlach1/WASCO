#!/bin/bash

#SBATCH --job-name=WASCO_script

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --nodelist=n141
#SBATCH --time=24:00:00
#SBATCH --partition=dept_cpu,camacho_gpu
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=None
#SBATCH --error=sbatch.%A.stderr
#SBATCH --output=sbatch.%A.stdout
#----PARAMETERS------------------------------------------------------------

#scratch drive folder to work in
JOBID=$(echo $SLURM_JOB_ID| grep -oP '^\d+')
SCRATCH_DIR=/scr/$USER/$SLURM_JOB_NAME/$JOBID

#if the scratch drive doesn't exist (it shouldn't) make it.
if [[ ! -e ${SCRATCH_DIR} ]]; then
    mkdir -p ${SCRATCH_DIR} && echo "scratch drive ${SCRATCH_DIR}"
fi

# change to scratch directory so job runs locally isntead of over the network
# (slows everyone down)
cd ${SCRATCH_DIR}

# copy files to working directory

$NAME1 = '1-3' 
$NAME2 = '2-4'
$RESULTS = 'slurm1' 

mkdir ensemble_1
mkdir ensemble_2

rsync -av ${SLURM_SUBMIT_DIR}/py_scripts/*.py ${SCRATCH_DIR} 
rsync -av ${SLURM_SUBMIT_DIR}/Examples/1FMK_peptides/ensemble_1/*.pdb ${SCRATCH_DIR}/ensemble_1/   
rsync -av ${SLURM_SUBMIT_DIR}/Examples/1FMK_peptides/ensemble_2/*.pdb ${SCRATCH_DIR}/ensemble_2/   


#copy files on exit or interrupt
trap clean_up EXIT

#define clean_up
clean_up(){
    echo 'copying files'
    rsync -av ${SCRATCH_DIR}/${RESULTS}/* ${SLURM_SUBMIT_DIR}

}
#-----Running Jobs --------

module load anaconda/3-cluster
source activate IDP-wcomp # this environment must exist already

python run_method.py --n1 $NAME1 --n2 $NAME2 --results $RESULTS

#--------------------------------------------------------------------------
# Leave this line to tell slurm that the script finished correctly
exit 0
