python train.py --config=configsYd/crack_resattentionnet_crossentr.json
python train.py --config=configsYd/crack_resattentionnet_bce.json
python train.py --config=configsYd/crack_resattentionnet_dice.json
python train.py --config=configsYd/crack_segnet_crossentr.json
python train.py --config=configsYd/crack_segnet_bce.json
python train.py --config=configsYd/crack_segnet_dice.json
python train.py --config=configsYd/crack_fusionnet_crossentr.json
python train.py --config=configsYd/crack_fusionnet_bce.json
python train.py --config=configsYd/crack_fusionnet_dice.json

python train.py --config=configsYd/crack_enet_bce.json
python train.py --config=configsYd/crack_enet_dice.json
python train.py --config=configsYd/crack_deeplabv3_crossentr.json
python train.py --config=configsYd/crack_deeplabv3_bce.json
python train.py --config=configsYd/crack_deeplabv3_dice.json

#python predict.py --config=configs/crackselected_linknet.json -id=2020-11-02-15-14-4953
