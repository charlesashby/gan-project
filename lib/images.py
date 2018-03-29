import numpy as np
import scipy.misc
import glob
import os
from PIL import Image

PATH = 'data/img_align_celeba'

img_test = '000001.jpg'
img_test_path = os.path.join(PATH, img_test)


def crop(image):
    image = scipy.misc.imresize(image, [128, 128])
    h, w = image.shape[:2]
    j = int(round((h - 64) / 2.))
    i = int(round((w - 64) / 2.))
    return image[j:j + 64, i: i + 64]


def save_images(images, name):
    # images are tensor of shape [b, 64, 64, 3]
    images = (images + 1.) * 127.5
    image = images[0].astype('uint8')
    image = Image.fromarray(image, 'RGB')
    image.save('./samples/batch-%s.jpg' % name)


def inverse_transform(images):
    return (images+1.)/2.


#def save_images(images, size, image_path):
 #   return imsave(inverse_transform(images), size, image_path)


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)


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
                              crop=True)
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