python -m torch.distributed.launch --nproc_per_node=4 --use_env train_multi_GPU.py --weights=''  --data=dataTSRDFull/TSRD.data --epochs=3000 --batch-size=12 --savebest=True --freeze-layers=False
