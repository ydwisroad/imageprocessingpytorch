python train.py --config=configsNew/crack_deeplabv3_bce.json
python train.py --config=configsNew/crack_enet_bce.json
python train.py --config=configsNew/crack_exfuse_bce.json
python train.py --config=configsNew/crack_fcn_bce.json
python train.py --config=configsNew/crack_segnet_bce.json

python predict.py --config=configs/crackselected_linknet.json -id=2020-11-02-15-14-4953
