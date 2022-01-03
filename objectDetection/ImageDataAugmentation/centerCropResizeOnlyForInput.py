import torchvision.transforms
import PIL.Image as Image
import torchvision.transforms
import os
import tqdm


def centerCropImages(src_images_dir, dest_images_dir, size=(640,640)):
    images = os.listdir(src_images_dir)

    for img_name in images:
        image =Image.open(src_images_dir + "/" + img_name)
        print(image.size, image.format, image.mode)

        width = image.size[0]
        height = image.size[1]
        eqSize = 512
        if (width > height):
            eqSize = height
        else:
            eqSize = width
        # 生成一个CenterCrop类的对象,用来将图片从中心裁剪成 size
        crop_obj = torchvision.transforms.CenterCrop((eqSize, eqSize))
        image = crop_obj(image)

        resize_obj = torchvision.transforms.Resize(size)
        image = resize_obj(image)

        # 将裁剪之后的图片保存下来
        image.save(dest_images_dir + "/" + img_name, format='PNG')

if __name__ == "__main__":
    src_images_dir = "E:/ubuntushare/data/bolt/original"
    dest_images_dir = "E:/ubuntushare/data/bolt/centercrop"
    if not os.path.exists(dest_images_dir):
        os.mkdir(dest_images_dir)
    centerCropImages(src_images_dir, dest_images_dir)

