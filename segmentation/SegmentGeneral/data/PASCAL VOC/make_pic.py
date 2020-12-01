from PIL import Image

train_path_txt = "./train.txt"
train_label_path_txt = "./train_labels.txt"

val_path_txt = "./val.txt"
val_label_path_txt = "./val_labels.txt"

train = "./train/"
train_label = "./train_labels/"

val = "./val/"
val_label = "./val_labels/"

path_file = open(val_label_path_txt, 'r')

for line in path_file.readlines():
    single_img = line.strip('\n')
    img = Image.open(single_img)
    img.save(val_label + single_img[-15:])

path_file.close()

