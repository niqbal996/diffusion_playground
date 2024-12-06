srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=10 -p A100-40GB --mem=50000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/mmsegmentation:/home/iqbal/mmsegmentation,/ds/images/cropandweed:/ds/images/cropandweed \
  --container-image=/netscratch/naeem/mmseg-torch23.04_updated.sqsh  \
  --container-workdir=/home/iqbal/mmsegmentation \
  --mail-type=END --mail-user=naeem.iqbal@dfki.de --job-name=syn_cropweed_baseline \
  --time=00-12:00 \
  bash train_syclops.sh