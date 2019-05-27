""" Split the handwritten math symbols dataset and resize images to 28x28.

The dataset comes in the following format:
	data/
		log/
			filename.jpg
			...
		sigma/
			filename.jpg
			...

We have to create train val and test sets
For a default it's 70% for train 20% for val and 10% for test
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE = 28
VAL_SIZE = 0.2
TEST_SIZE = 0.1
TRAIN_SIZE = 1. - VAL_SIZE - VAL_SIZE

parser =  argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory with handwritten math symbols dataset")
parser.add_argument('--output_dir', default='data_split', help="Where to write the new daata")
parser.add_argument('--rename_images', default=False, help="Rename all images")


def resize_and_save(filename, output_dir, size=SIZE):
	image = Image.open(filename)
	# Use bilinear interpolation instead of the default "nearest neigbor" method
	image = image.resize((size, size), Image.BILINEAR)
	image.save(os.path.join(output_dir, filename.split('/')[-1]))


def rename_images(data_dir):
    '''rename_images func take list of labels as argument and rename
    all images in subdirectories to {label_name}_{i}.jpg formula for
    each one  
    '''
    label_names = os.listdir(data_dir)
    for label_name in label_names:
        label_path = os.path.join(data_dir, label_name)
        filenames = os.listdir(label_path)
        filenames = [os.path.join(label_path, f) for f in filenames]
        for i, filename in enumerate(filenames):
            new_ending = label_name + '_' + str(i)
            new_filename = filename.replace(filename.split('/')[-1], new_ending)
            os.rename(filename, new_filename)


def split_train_val_test(data_dir, output_dir):
	label_names = os.listdir(data_dir)
	for label_name in label_names:
		label_path = os.path.join(data_dir, label_name)

		train_output_path = os.path.join(output_dir, 'train')
		train_output_label_path = os.path.join(train_output_path, label_name)

		val_output_path = os.path.join(output_dir, 'val')
		val_output_label_path = os.path.join(val_output_path, label_name)
		
		test_output_path = os.path.join(output_dir, 'test')
		test_output_label_path = os.path.join(test_output_path, label_name)
		

		filenames = os.listdir(label_path)
		filenames = [os.path.join(label_path, f) for f in filenames]
		
		filenames.sort()
		random.shuffle(filenames)

		split0 = int(TRAIN_SIZE * len(filenames)) 
		split1 = int((TRAIN_SIZE+VAL_SIZE) * len(filenames))

		train_filenames = filenames[:split0]
		val_filenames = filenames[split0:split1]
		test_filenames = filenames[split1:]

		if not os.path.exists(train_output_label_path):
			os.mkdir(train_output_label_path)
		else:
			print('Warning dir {} already exists'.format(train_output_label_path))

		if not os.path.exists(val_output_label_path):
			os.mkdir(val_output_label_path)
		else:
			print('Warning dir {} already exists'.format(val_output_label_path))

		if not os.path.exists(test_output_label_path):
			os.mkdir(test_output_label_path)
		else:
			print('Warning dir {} already exists'.format(test_output_label_path))


		for f in train_filenames:
			resize_and_save(f, train_output_label_path)

		for f in val_filenames:
			resize_and_save(f, val_output_label_path)

		for f in test_filenames:
			resize_and_save(f, test_output_label_path)


	



if __name__ == '__main__':
	args = parser.parse_args()

	assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

	random.seed(230)

	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)
	else:
		print("Warning output_dir {} already exists".format(args.output_dir))

	if args.rename_images:
		rename_images(args.data_dir)

	split_train_val_test(args.data_dir, args.output_dir)



