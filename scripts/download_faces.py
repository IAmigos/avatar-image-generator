import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm_n
from tqdm import tqdm
from time import time
import cv2
from keras_segmentation.pretrained import pspnet_101_voc12

# !pip install keras-segmentation


def read_file(path, min_pose, min_score, curation, formats_allowed):
    ''' Reads a text file and returns info of each image '''

    with open(path, 'r') as in_file:
        for line in in_file:
            # url
            # bbox coordinates
            # pose: frontal/profile (pose>2 signifies a frontal face while pose<=2 represents left and right profile detection)
            # detection score: Score of a DPM detector
            # curation: Whether this image was a part of final curated dataset (1 or 0)
            _, url, x1, y1, x2, y2, pose, score, curated_dataset = line.split(
                ' ')
            if url.split('.')[-1] in formats_allowed and float(pose) >= min_pose:
                # Same as: not curation or (curation and int(curated_dataset))
                if not curation or int(curated_dataset):
                    yield url, [x1, y1, x2, y2]
    return []


def download_crop_image(item, offset_x, offset_top_percent, offset_bottom_percent):
    url, bbox = item
    x1, y1, x2, y2 = list(map(float, bbox))

    im_cropped = None
    try:
        response = requests.get(url, stream=True)
        response.raw.decode_content = True
        im = Image.open(response.raw)
        w = (x2 - x1)
        h = (y2 - y1)
        v_offset_x = w*offset_x/100
        v_offset_top = h*offset_top_percent/100
        v_offset_bottom = h*offset_bottom_percent/100
        im_cropped = im.crop(
            (x1-v_offset_x, y1-v_offset_top, x2+v_offset_x, y2+v_offset_bottom))
    except OSError:
        # tqdm.write('404: '+url)
        pass

    return im_cropped


def get_image(person_name, file_w_path, num_images, target_path, offset_x,
              offset_top_percent, offset_bottom_percent, min_pose, min_score,
              curation, formats_allowed):
    ''' Downloads images from a given text file, crops them and saves them locally '''

    dir_path = target_path  # os.path.dirname(target_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    i_image = 0
    for item in read_file(file_w_path, min_pose, min_score, curation, formats_allowed):
        if i_image >= num_images:
            # tqdm.write('Finished Downloading '+person_name)
            return
        image = download_crop_image(
            item, offset_x, offset_top_percent, offset_bottom_percent)
        if image:
            try:
                image.save(os.path.join(
                    dir_path, f'{person_name}_{i_image+1}.jpg'))
            except Exception:
                continue
            i_image += 1
    tqdm.write(f'There is no more images available for {person_name}')
    tqdm.write(f'({i_image} image(s) downloaded of {num_images})\n')


def download_vgg_images(data_path, num_people, num_images, target_path, offset_x_percent,
                        offset_top_percent, offset_bottom_percent, min_pose=3, min_score=0,
                        curation=False, formats_allowed=['jpg', 'jpeg'], from_notebook=False):
    ''' 
        Parses info of text files in a given folder and downloads the images 
        in them. Each file contains info about images of the same person.
    '''

    ini = time()

    target_path += '/'
    files = os.listdir(data_path)
    n = num_people if num_people < len(files) else len(files)
    pbar_class = tqdm_n if from_notebook else tqdm
    pbar = pbar_class(files[:num_people], total=n)
    for fi in pbar:
        # The file's name is the name of the person
        fi_splited = fi.split('.')
        person_name = '.'.join(fi_splited[:-1]) if len(fi_splited) > 1 else fi
        pbar.set_description(
            f'Processing {person_name} ({num_images} image(s))')
        file_w_path = data_path + '/' + fi
        get_image(person_name, file_w_path, num_images,
                  target_path, offset_x_percent, offset_top_percent,
                  offset_bottom_percent, min_pose, min_score, curation,
                  formats_allowed)

    end = time()
    print(f'Downloading time: {round((end-ini)/60, 2)} min\n')


def clean_corrupt_files(path, formats_allowed=['jpg', 'jpeg']):
    ''' Filters files that have only one channel, can't be opened or have a format not allowed '''

    n_removed = 0
    for filename in os.listdir(path):
        filename_lower = filename  # .lower()
        if filename_lower.split('.')[-1] in formats_allowed:
            try:
                img = Image.open(
                    os.path.join(path, filename))  # open the image file
                img.verify()  # verify that it is in fact an image
                if len(plt.imread(path + filename).shape) != 3:
                    os.remove(os.path.join(path, filename))
                    print(f'Removing corrupt file:', filename)
                    n_removed += 1
            except Exception:
                os.remove(os.path.join(path, filename))
                print('Removing corrupt file:', filename)
                n_removed += 1
        else:
            os.remove(os.path.join(path, filename))
            print('Removing file with different format:', filename)
            n_removed += 1
    print(f'\n{n_removed} file(s) removed')


def remove_background(images_path, masks_path, output_path, from_notebook=False):
    ''' Removes background from images (paints it white) '''

    if not os.path.exists(masks_path):
        os.makedirs(masks_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load the pretrained model trained on Pascal VOC 2012 dataset
    model = pspnet_101_voc12()

    pbar_class = tqdm_n if from_notebook else tqdm
    pbar = pbar_class(os.listdir(images_path))
    start = time()
    for filename in pbar:
        pbar.set_description(f'Processing {filename}')
        mask_file = filename[:-4] + "_mask.png"
        output_file = filename[:-4] + "_wo_bg.jpg"

        model.predict_segmentation(
            inp=images_path + filename,
            out_fname=masks_path + mask_file
        )

        img_mask = cv2.imread(masks_path + mask_file)
        img1 = cv2.imread(images_path + filename)  # READ BGR

        seg_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        _, bg_mask = cv2.threshold(
            seg_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # convert mask to 3-channels
        bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)

        # cv2.bitwise_or to extract the region
        bg = cv2.bitwise_or(img1, bg_mask)

        # save
        cv2.imwrite(output_path + output_file, bg)
    end = time()

    print('Background removed')
    print(f'Processing time: {round((end-start)/60, 2)} min\n')


if __name__ == "__main__":
    data_path = r'C:\Users\Daniel Ibáñez\Documents\Proyectos\Avatar Project\vgg_face_dataset\files'
    target_path = r'C:\Users\Daniel Ibáñez\Documents\Proyectos\Avatar Project\prueba/'

    download_vgg_images(data_path, num_people=50, num_images=3, target_path=target_path,
                        offset_x_percent=15, offset_top_percent=55,
                        offset_bottom_percent=12, min_pose=3, min_score=0,
                        curation=False, formats_allowed=['jpg', 'jpeg'], from_notebook=False)
    clean_corrupt_files(target_path, formats_allowed=['jpg', 'jpeg'])

    images_path = r'C:\Users\Daniel Ibáñez\Documents\Proyectos\Avatar Project\tmp\face_images/'
    masks_path = r'C:\Users\Daniel Ibáñez\Documents\Proyectos\Avatar Project\tmp\masks/'
    output_path = r'C:\Users\Daniel Ibáñez\Documents\Proyectos\Avatar Project\face_images_wo_bg/'

    remove_background(images_path, masks_path,
                      output_path, from_notebook=False)
