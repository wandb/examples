from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator

def adjust_image(img):
    return img / 255


def adjust_mask(mask):
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask


def get_training_data(batch_size, train_path, image_folder, mask_folder,
    aug_dict, target_size, seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode="rgb",
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)
    for img, mask in train_generator:
        yield adjust_image(img), adjust_mask(mask)
