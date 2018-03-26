import numpy as np
import scipy.misc
import glob
import os
from PIL import Image

PATH = 'data/img_align_celeba'

img_test = '000001.jpg'
img_test_path = os.path.join(PATH, img_test)


def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=False, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(
        x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
              image, input_height, input_width,
              resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.


def visualize(image):
    img = image.astype('uint8')
    img = Image.fromarray(img, 'RGB')
    img.show()


def load_data(images, batch_size, batch_index, split='train'):

    if split == 'train':
        batch_imgs = images[batch_index * batch_size:
                            (batch_index + 1) * batch_size]
    elif split == 'valid':
        batch_imgs = images[batch_index * batch_size + 170000:
                            (batch_index + 1) * batch_size + 170000]
    else:
        batch_imgs = images[batch_index * batch_size + 200000:
                            (batch_index + 1) * batch_size + 200000]

    for i, image in enumerate(batch_imgs):
        img = Image.open(image)
        img_array = np.array(img)
        img_array = transform(img_array, input_height=img_array.shape[0],
                              input_width=img_array.shape[1],
                              crop=False)
        img_array = np.expand_dims(img_array, axis=0)

        if i == 0:
            y = img_array
        else:
            y = np.concatenate((y, img_array), axis=0)
    return np.random.normal(size=(batch_size, 100)).astype('float32'), y.astype('float32')


def iterate_minibatches(batch_size, split='train'):
    images = glob.glob(PATH + "/*.jpg")
    #l = len(images)
    l = 170000
    for idx in range(0, l // batch_size):

        z, targets = load_data(images, batch_size, idx, split=split)

        yield z, targets