from toolbox.datasets.camvid import CamVid


def get_dataset(cfg):

    crop_size = (cfg['image_h'], cfg['image_w'])
    num_class = cfg['n_classes']

    if cfg['dataset'] == 'camvid':
        class_dict_path = './database/camvid/class_dict.csv'
        TRAIN_ROOT = './database/camvid/train'
        TRAIN_LABEL = './database/camvid/train_labels'
        VAL_ROOT = './database/camvid/val'
        VAL_LABEL = './database/camvid/val_labels'
        TEST_ROOT = './database/camvid/test'
        TEST_LABEL = './database/camvid/test_labels'

        TRAIN_GRAY = './database/camvid/trainannot'
        return CamVid([TRAIN_ROOT, TRAIN_LABEL, class_dict_path, TRAIN_GRAY], crop_size, num_class), \
               CamVid([VAL_ROOT, VAL_LABEL, class_dict_path, TRAIN_GRAY], crop_size, num_class), \
               CamVid([TEST_ROOT, TEST_LABEL, class_dict_path, TRAIN_GRAY], crop_size, num_class)
    else:
        return
