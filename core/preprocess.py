#!usr/bin/python
import os
import numpy as np
import cv2
from PIL import ImageOps
import random
from copy import deepcopy
import matplotlib.pyplot as plt

NUM_DIR = 1


def imshow(img):
    plt.imshow(img)
    plt.show()


def get_data(num_dir=NUM_DIR, data_dir='../data/'):
    # gives CV2 format
    originals = []
    masks = []
    dirnames = np.arange(num_dir)
    for dirname in dirnames:
        dirname = data_dir + str(dirname)
        # print(dirname)
        if os.path.isdir(data_dir + str(dirname)):
            for i, filename in enumerate(sorted(os.listdir(str(dirname)))):
                if (filename != '.DS_Store'):
                    name = filename.split('.')[0]
                    if (name[-5:] == 'imcor'):
                        img = cv2.imread(str(dirname) + '/' + str(filename))
                        originals.append(img)
                    elif (name[-3:] == 'bin'):
                        img = cv2.imread(str(dirname) + '/' + str(filename), 0)
                        masks.append(img)
                    else:
                        pass
    return originals, masks


# deprecated
def pad(img, crop_size=500):
    # takes and returns PIL format
    width, height = img.size
    desired_size = [width // crop_size, height // crop_size]

    delta_w = desired_size[0] - new_size[0]
    delta_h = desired_size[1] - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

    padded_img = ImageOps.expand(img, padding)

    return padded_img


def resize_img(img, crop_size=500):
    # takes, returns CV2 format
    height, width = img.shape[0], img.shape[1]
    desired_size = (crop_size * (width // crop_size), crop_size * (height // crop_size))
    return cv2.resize(img, desired_size)


def resized_only_wear(originals, masks, crop_size=500):
    # gives CV2 format
    preprocessed = []
    for (ori_t, mask_t) in zip(originals, masks):
        wear_img = cv2.bitwise_and(ori_t, ori_t, mask=mask_t)
        # preprocessed.append(pad(Image.fromarray(wear_img)),crop_size)	#convert to PIL before passing pad
        preprocessed.append(resize_img(wear_img, crop_size))

    return preprocessed


def resize_original(originals, crop_size=500):
    # gives CV2 format
    preprocessed = []
    for img in originals:
        preprocessed.append(resize_img(img, crop_size))

    # pdb.set_trace()
    return preprocessed


def cutout(raw_image, contour, crop_size=500, threshold=0.2):
    # imshow(contour[1])
    # takes cv2 format # return preprocessed cutouts which is list of list of [cutouts,location_of_cutout]
    preprocessed_cutouts = []  # each element is [img,corresponding_contour,wear_value]
    wear_cut = []
    no_wear = []
    wear_count = 0  # total num of cutouts with high wear(wear value>0.9)
    test_flag = 0
    # case when test images
    if (len(contour) == 0):
        contour = range(len(raw_image))
        test_flag = 1

    for (img, cnt) in zip(raw_image, contour):
        cuts = []
        cuts_imgdata = []
        hei = img.shape[0]
        wid = img.shape[1]
        num_x = wid // crop_size
        num_y = hei // crop_size
        for x in range(num_x):
            for y in range(num_y):
                cuts.append([x * crop_size, y * crop_size, (x + 1) * crop_size, (y + 1) * crop_size])
                cuts_imgdata.append([img[y * crop_size:(y + 1) * crop_size, x * crop_size:(x + 1) * crop_size],
                                     [x * crop_size, (x + 1) * crop_size, y * crop_size, (y + 1) * crop_size]])

        # pdb.set_trace()

        preprocessed_cutouts.append(cuts_imgdata)

        if (test_flag == 0):
            for cut_dims in cuts:
                x1, y1, x2, y2 = cut_dims
                normed_cnt_cutout = cnt[y1:y2, x1:x2] / 255.0
                wear_value = np.sum(normed_cnt_cutout) / (normed_cnt_cutout.shape[0] * normed_cnt_cutout.shape[1])
                if (wear_value > threshold):
                    wear_count += 1
                current_cutout = img[y1:y2, x1:x2]
                # preprocessed_cutouts.append([current_cutout,normed_cnt_cutout*255,wear_value])
                if (threshold < wear_value):
                    # print("Wear with wear_value = ", wear_value)
                    wear_cut.append([current_cutout, normed_cnt_cutout, wear_value])
                else:
                    # print("No wear with wear_value = ", wear_value)
                    no_wear.append([current_cutout, normed_cnt_cutout, wear_value])

    print(wear_count, "number of high wear cutouts were identified")

    # pdb.set_trace()

    return wear_cut, no_wear, preprocessed_cutouts


def process(crop_size=128, threshold=0.5, data_dir='../data/', num_dir=NUM_DIR):
    print("Preprocessing data...")
    originals, masks = get_data(num_dir, data_dir)

    resized_originals = resize_original(originals, crop_size=crop_size)

    resized_masks = resize_original(masks, crop_size=crop_size)
    # resized_wear = resized_only_wear(originals, masks, crop_size)		#cv2 format
    wear_cut, no_wear_cut, preprocessed_cutouts = cutout(resized_originals, resized_masks, crop_size=crop_size,
                                                         threshold=threshold)
    # pdb.set_trace()
    print("{} images preprocessed, {} preprocessed cutouts formed".format(len(preprocessed_cutouts),
                                                                          (len(wear_cut) + len(no_wear_cut))))

    original_shape = []
    for original in originals:
        original_shape.append(original.shape)

    return wear_cut, no_wear_cut, preprocessed_cutouts, original_shape


""" TODO: Implement bias_ratio """


def Imdb(wear_cut, no_wear_cut, bias_ratio=0.5, no_cut_select=0.99):
    """ bias_ratio is bias of wear to no_wear i.e. bias_ratio*batch_size=#wear images"""
    num_to_select = int(no_cut_select * len(no_wear_cut))
    no_wear_cut = random.sample(no_wear_cut, num_to_select)

    data_list = [wear_cut, no_wear_cut]

    sizes = [len(wear_cut), len(no_wear_cut)]

    batch_size = deepcopy(np.amax(sizes))
    index_to_update = deepcopy(np.argmin(sizes))

    list_to_update = deepcopy(data_list[index_to_update])  # this list doesnt change
    updated_list = deepcopy(list_to_update)

    for i in range((batch_size - len(list_to_update)) // len(list_to_update)):
        updated_list += list_to_update
    # add remaining data here

    trailing_size = batch_size - len(updated_list)
    trailing_idx = random.sample(range(0, len(list_to_update)), trailing_size)
    updated_list += [list_to_update[idx] for idx in trailing_idx]

    data_list[np.argmin(sizes)] = updated_list

    [wear_cut, no_wear_cut] = data_list

    assert len(wear_cut) == batch_size
    assert len(no_wear_cut) == batch_size

    batch_images = [x1[0] for x1 in wear_cut] + [x2[0] for x2 in no_wear_cut]
    batch_labels = [x1[1] for x1 in wear_cut] + [x2[1] for x2 in no_wear_cut]

    """WARNING: only for grey images """
    # for i in range(len(batch_images)):
    #     batch_images[i] = np.expand_dims(batch_images[i], axis=0)
    """WARNING: only for BGR images """
    for i in range(len(batch_images)):
        batch_images[i] = batch_images[i].transpose([2, 0, 1])
    #
    # assert len(batch_images) == 2 * batch_size
    # assert len(batch_labels) == 2 * batch_size

    print("Data loaded!")
    return batch_images, batch_labels


def main():
    threshold = 0.5
    wear_cut, no_wear_cut, _, _ = process()
    Imdb(wear_cut, no_wear_cut)


if __name__ == '__main__':
    main()
