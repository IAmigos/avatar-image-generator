import os
import requests
from PIL import Image
import pdb
from tqdm import *


class NotFound(BaseException):
	pass

def gmkdir(path):
  """ creates directory with class name if not present """

  if not os.path.exists(path):
    os.makedirs(path)

def clean(base_name):
  """ Removes txt extension from file name """

  return base_name.split('.')[0]

def read_file(path):
  """ Reads each text file and returns the url and bbox co-ordinates for each class"""

  def get_(line):
    _, url, x1, y1, x2, y2, _, _ ,_ = line.split(' ')
    return url, [x1, y1, x2, y2]
  with open(path, 'r') as in_file:
    lines = in_file.readlines()
  return [get_(x) for x in lines]

def get_image(data_path, key, items, num_images, target_path, offset_x, offset_y):
  """ Downloads the image from a given url , crops it and saves it locally"""

  ##dir_path = os.path.join(os.path.dirname(data_path), 'faces') + '/' + key
  ##dir_path = os.path.join(os.path.dirname(data_path), name_target_file)
  dir_path = os.path.dirname(target_path)

  gmkdir(dir_path)
  im_cropped = None

  def crop_image(item):
    url, bbox = item
    x1, y1, x2, y2 = list(map(float, bbox))

    try:
      response = requests.get(url, stream=True)
      response.raw.decode_content = True
      im = Image.open(response.raw)
      im_cropped = im.crop((x1-offset_x, y1-offset_y, x2+offset_x, y2+offset_y))
      return im_cropped
    except NotFound:
      print('404')
    except OSError:
      pass

  for i, item in enumerate(items):
    if i >= num_images:
      print('Finished Downloading')
      break
    image = crop_image(item)
    if image:
      try:
        ##image.save(os.path.join(target_path,
        ##            '%d.jpg'%i))

        image.save(os.path.join(dir_path, '{}_{}.jpg'.format(key,i)))


      except Exception as e:
        pass

def parse_data(data_path, num_ppl, num_images, target_path, offset_x, offset_y):
  content = {}
  dirs = lambda f: data_path +'/'+ f
  files = os.listdir(data_path)
  paths = map(dirs, files)
  for path in paths:
    content[clean(os.path.basename(path))] = read_file(path)

  keys = [key for key in content.keys()]
  pbar = tqdm(keys[:num_ppl])
  for key in pbar:
    pbar.set_description("Processing %s" % key)
    get_image(data_path, key, content[key], num_images, target_path, offset_x, offset_y)
