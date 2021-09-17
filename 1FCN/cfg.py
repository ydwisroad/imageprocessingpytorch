BATCH_SIZE = 2
EPOCH_NUMBER = 10
#DATASET = ['CamVid', 12]
#DATASET = ['PASCAL VOC', 21]
DATASET = ['crackSegmentationSimple', 2]

crop_size = (512, 512)

class_dict_path = '../../data/' + DATASET[0] + '/class_dict.csv'
TRAIN_ROOT = '../../data/' + DATASET[0] + '/train'
TRAIN_LABEL = '../../data/' + DATASET[0] + '/train_labels'
VAL_ROOT = '../../data/' + DATASET[0] + '/val'
VAL_LABEL = '../../data/' + DATASET[0] + '/val_labels'

RESULT_ROOT='../../data/result'

if DATASET == "PASCAL VOC":
    TEST_ROOT = None
    TEST_LABEL = None
else:
    TEST_ROOT = '../../data/' + DATASET[0] + '/test'
    TEST_LABEL = '../../data/' + DATASET[0] + '/test_labels'



