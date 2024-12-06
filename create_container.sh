srun -K --ntasks=1 --gpus-per-task=1 -N 1 --cpus-per-gpu=20 -p A100-40GB --mem=80000 \
  --container-mounts=/netscratch/naeem:/netscratch/naeem,/home/iqbal/diffusion_playground:/home/iqbal/diffusion_playground,/ds/images/cropandweed:/ds/images/cropandweed \
  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.10-py3.sqsh \
  --container-save=/netscratch/naeem/diffusion_pg_23.10.sqsh \
  --container-workdir=/home/iqbal/diffusion_playground \
  --time=00-02:00 \
  --immediate=300 \
  --pty /bin/bash
