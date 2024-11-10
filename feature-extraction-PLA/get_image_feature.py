#!/usr/bin/env python3
from argparse import ArgumentParser
from importlib import import_module
from itertools import count
import os
import glob
import json
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from aggregators import AGGREGATORS
import common

'''
example:
python get_image_feature.py  --experiment_root ./models_vehicle --checkpoint checkpoint-102873 --filename query/huilue999_16WW_car_05815_003_0322_0058_0372_0110.jpg
'''



parser = ArgumentParser(description='Embed a dataset using a trained network.')

# Required

parser.add_argument(
    '--experiment_root', default='./models_vehicle',
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--filename', default=None,
    help=r'''filename of the image to extract feature, path relative to ./Huilue/dataset, default= ['query/huilue999_16WW_car_05815_003_0322_0058_0372_0110.jpg']''')


# Optional

parser.add_argument(
    '--checkpoint', default=None,
    help='Name of checkpoint file of the trained network within the experiment '
         'root. Uses the last checkpoint if not provided.')

parser.add_argument(
    '--batch_size', default=256, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on available memory.')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)


def main(data_fids):
    # Verify that parameters are set correctly.
    args = parser.parse_args()

    # Load the args from the original experiment.
    args_file = os.path.join(args.experiment_root, 'args.json')

    if os.path.isfile(args_file):
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)

        # Add arguments from training.
        for key, value in args_resumed.items():
            args.__dict__.setdefault(key, value)

        args.image_root = args.image_root or args_resumed['image_root']
    else:
        raise IOError('`args.json` could not be found in: {}'.format(args_file))


    # Load the data from the CSV file.
    #_, data_fids = common.load_dataset(args.dataset, args.image_root)
    # if vars(args).get('filename',None) is None:
    #     #data_fids = ['query/huilue999_16WW_car_05815_003_0322_0058_0372_0110.jpg']
    #     data_fids = sorted(glob.glob('Huilue/reid_feature1/0_images_0_1469/*image*.jpg'))
    # else:
    #     data_fids = [vars(args)['filename']]

    net_input_size = (args.net_input_height, args.net_input_width)
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width)

    # Setup a tf Dataset containing all images.
    dataset = tf.data.Dataset.from_tensor_slices(data_fids)

    # Convert filenames to actual image tensors.
    dataset = dataset.map(
        lambda fid: common.fid_to_image(
            fid, 'dummy', image_root=args.image_root,
            image_size=pre_crop_size if args.crop_augment else net_input_size),
        num_parallel_calls=args.loading_threads)

    # Group it back into PK batches.
    dataset = dataset.batch(args.batch_size)

    # Overlap producing and consuming.
    dataset = dataset.prefetch(1)

    images, _, _ = dataset.make_one_shot_iterator().get_next()

    # Create the model and an embedding head.
    model = import_module('nets.' + args.model_name)
    head = import_module('heads.' + args.head_name)

    endpoints, body_prefix = model.endpoints(images, is_training=False)
    with tf.name_scope('head'):
        endpoints = head.head(endpoints, args.embedding_dim, is_training=False)

    with tf.Session() as sess:
        # Initialize the network/load the checkpoint.
        if args.checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(args.experiment_root)
        else:
            checkpoint = os.path.join(args.experiment_root, args.checkpoint)
        tf.train.Saver().restore(sess, checkpoint)

        # Go ahead and embed the whole dataset, with all augmented versions too.
        emb_storage = np.zeros((len(data_fids), args.embedding_dim), np.float32)
        for start_idx in count(step=args.batch_size):
            try:
                emb = sess.run(endpoints['emb'])
                print('\rEmbedded batch {}-{}/{}'.format(start_idx, start_idx + len(emb), len(emb_storage)))
                emb_storage[start_idx:start_idx + len(emb)] = emb
            except tf.errors.OutOfRangeError:
                break  # This just indicates the end of the dataset.


        for i in range(emb_storage.shape[0]):
            feature = emb_storage[i] / np.linalg.norm(emb_storage[i], ord=2)
            emb_storage[i,...] = feature
        return emb_storage

if __name__ == '__main__':

    vehicle_fids = sorted(glob.glob(os.path.join('query','*.jpg')))

    tf.reset_default_graph()
    Feature = main(vehicle_fids)
    
    feature_filename = 'query_descriptors.pkl'
    joblib.dump(Feature,feature_filename,compress=9)

    #print Feature
