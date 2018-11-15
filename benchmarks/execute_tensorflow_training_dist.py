#!/usr/bin/env python

from __future__ import absolute_import

import argparse
import itertools
import os

from sagemaker import Session
from sagemaker.tensorflow import TensorFlow
import datetime

default_bucket = Session().default_bucket
dir_path = os.path.dirname(os.path.realpath(__file__))

_DEFAULT_HYPERPARAMETERS = {
    'num_batches':           1000,
    'model':                'vgg16',
    'batch_size':           64,
    'summary_verbosity':    1,
    'save_summaries_steps': 10,
    'data_name':            'imagenet'
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--instance-types', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-n', '--instance-counts', nargs='+', help='<Required> Number of instances for training', required=True, type=int)
    parser.add_argument('-ps', '--ps-distribution', default= False,
                        help='Enable parameter server based distributed training',
                        action='store_true')
    parser.add_argument('-r', '--role', required=True)
    parser.add_argument('-w', '--wait', action='store_true')
    parser.add_argument('--region', default='us-west-2')
    parser.add_argument('--py-versions', nargs='+', help='<Required> Set flag', default=['py3'])
    parser.add_argument('--checkpoint-path',
                        default=os.path.join(default_bucket(), 'benchmarks', 'checkpoints'),
                        help='The S3 location where the model checkpoints and tensorboard events are saved after training')

    return parser.parse_known_args()


def main(args, script_args):
    for instance_count, instance_type, py_version in itertools.product(args.instance_counts, args.instance_types, args.py_versions):
        base_name = '%s-%s-%s-%sx-%s' % (py_version, instance_type[3:5], instance_type[6:], instance_count,str(datetime.datetime.now()).replace(' ','-').replace(':','-').replace('.','-'))
        model_dir = os.path.join(args.checkpoint_path, base_name)

        output_dir = "s3://tf-benchmarks-us-west-2/output/"

        job_hps = create_hyperparameters(model_dir, script_args)

        print('hyperparameters:')
        print(job_hps)

        estimator = TensorFlow(entry_point='tf_cnn_benchmarks.py',
                               source_dir=os.path.join(dir_path, 'tf_cnn_benchmarks'),
                               role='SageMakerRole',
                               train_instance_count=instance_count,
                               train_instance_type=instance_type,
                               framework_version='1.11',
                               py_version='py3',
                               distributions={'parameter_server': {'enabled': True}},
                               hyperparameters=job_hps,
                               base_job_name=base_name,
                               output_path=output_dir)

        print(estimator.train_image())

        estimator.fit(wait=args.wait)

    print("To use TensorBoard, execute the following command:")
    cmd = 'S3_USE_HTTPS=0 S3_VERIFY_SSL=0  AWS_REGION=%s tensorboard --host localhost --port 6006 --logdir %s'
    print(cmd % (args.region, args.checkpoint_path))




def create_hyperparameters(model_dir, script_args):
    job_hps = _DEFAULT_HYPERPARAMETERS.copy()

    job_hps.update({'train_dir': model_dir, 'eval_dir': model_dir})

    script_arg_keys_without_dashes = [key[2:] if key.startswith('--') else key[1:] for key in script_args[::2]]
    script_arg_values = script_args[1::2]
    job_hps.update(dict(zip(script_arg_keys_without_dashes, script_arg_values)))

    return job_hps


if __name__ == '__main__':
    args, script_args = get_args()
    print(str(args))
    main(args, script_args)
