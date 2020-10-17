from toolbox.datasets.segmentdataset import SegmentDataset

def get_dataset(cfg):

    crop_size = (cfg['image_h'], cfg['image_w'])
    num_class = cfg['n_classes']

    #if cfg['dataset'] == 'camvid':
    class_dict_path = './' + cfg['datasetpath'] + '/' + cfg['dataset'] + '/class_dict.csv'
    TRAIN_ROOT = './' + cfg['datasetpath'] + '/' + cfg['dataset'] + '/train'
    TRAIN_LABEL = './' + cfg['datasetpath'] + '/' + cfg['dataset'] + '/train_labels'
    VAL_ROOT = './' + cfg['datasetpath'] + '/' + cfg['dataset'] + '/val'
    VAL_LABEL = './' + cfg['datasetpath'] + '/' + cfg['dataset'] + '/val_labels'
    TEST_ROOT = './' + cfg['datasetpath'] + '/' + cfg['dataset'] + '/test'
    TEST_LABEL = './' + cfg['datasetpath'] + '/' + cfg['dataset'] + '/test_labels'

    TRAIN_GRAY = './' + cfg['datasetpath'] + '/' + cfg['dataset'] + '/trainannot'
    return SegmentDataset([TRAIN_ROOT, TRAIN_LABEL, class_dict_path, TRAIN_GRAY], crop_size, num_class), \
               SegmentDataset([VAL_ROOT, VAL_LABEL, class_dict_path, TRAIN_GRAY], crop_size, num_class)   #, \
               #SegmentDataset([TEST_ROOT, TEST_LABEL, class_dict_path, TRAIN_GRAY], crop_size, num_class)

