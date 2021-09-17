python -m torch.distributed.launch --nproc_per_node=4 --use_env train_multi_GPU.py --weights=''  --data=dataTSRD/TSRD.data --epochs=5 --batch-size=12 --savebest=True --freeze-layers=False
