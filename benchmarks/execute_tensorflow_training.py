#!/usr/bin/env python

from __future__ import absolute_import

import contextlib
import itertools
import os
import shutil
import subprocess
import tempfile

import click

from sagemaker import Session
from sagemaker.tensorflow import TensorFlow

default_bucket = Session().default_bucket()
dir_path = os.path.dirname(os.path.realpath(__file__))


@click.group()
def cli():
    pass


@cli.command('train')
@click.option('--framework-version', required=True, type=click.Choice(['1.11.0', '1.12.0']))
@click.option('--device', required=True, type=click.Choice(['cpu', 'gpu']))
@click.option('--py-versions', multiple=True, type=str)
@click.option('--training-input-mode', default='File', type=click.Choice(['File', 'Pipe']))
@click.option('--networking-isolation/--no-networking-isolation', default=False)
@click.option('--wait/--no-wait', default=False)
@click.option('--security-groups', multiple=True, type=str)
@click.option('--subnets', multiple=True, type=str)
@click.option('--role', default='SageMakerRole', type=str)
@click.option('--instance-counts', multiple=True, type=int)
@click.option('--batch-sizes', multiple=True, type=int)
@click.option('--instance-types', multiple=True, type=str)
@click.argument('script_args', nargs=-1, type=str)
def train(framework_version,
          device,
          py_versions,
          training_input_mode,
          networking_isolation,
          wait,
          security_groups,
          subnets,
          role,
          instance_counts,
          batch_sizes,
          instance_types,
          script_args):

    iterator = itertools.product(instance_types, py_versions, instance_counts, batch_sizes)
    for instance_type, py_version, instance_count, batch_size in iterator:
        base_name = job_name(instance_type, instance_count, device, py_version, batch_size)

        template = """#!/usr/bin/env bash 
        pip install requests py-cpuinfo psutil

        PYTHONPATH="/opt/ml/code/models:$PYTHONPATH" python benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py %s \
        --train_dir /opt/ml/model --eval_dir /opt/ml/model --benchmark_log_dir /opt/ml/model --batch_size %s\n"""

        script = template % (' '.join(script_args), batch_size)

        print('Creating laucher.sh:\n')
        print(script)

        with _tmpdir() as tmp:
            entry_point = os.path.join(tmp, 'launcher.sh')
            with open(entry_point, mode='w') as f:
                f.write(script)

            estimator = TensorFlow(
                entry_point=entry_point,
                role=role,
                dependencies=[os.path.join(dir_path, 'benchmarks'), os.path.join(dir_path, 'models')],
                base_job_name=base_name,
                train_instance_count=instance_count,
                train_instance_type=instance_type,
                framework_version=framework_version,
                py_version=py_version,
                script_mode=True,
                security_group_ids=security_groups,
                subnets=subnets
            )

            estimator.fit(wait=wait)

            if wait:
                artifacts_path = os.path.join(dir_path, 'results', estimator.latest_training_job.job_name)
                model_path = os.path.join(artifacts_path, 'model.tar.gz')
                os.makedirs(artifacts_path)
                subprocess.call(['aws', 's3', 'cp', estimator.model_data, model_path])
                subprocess.call(['tar', '-xvzf', model_path], cwd=artifacts_path)

                print('Model downloaded at %s' % model_path)


def job_name(instance_type,
             instance_count,
             device,
             python_version,
             batch_size):
    instance_typename = instance_type.replace('.', '').replace('ml', '')

    return 'tf-%s-%s-%s-%s-b%s' % (instance_typename, instance_count, device, python_version, batch_size)


@contextlib.contextmanager
def _tmpdir(suffix='', prefix='tmp', dir=None):  # type: (str, str, str) -> None
    """Create a temporary directory with a context manager. The file is deleted when the context exits.

    The prefix, suffix, and dir arguments are the same as for mkstemp().

    Args:
        suffix (str):  If suffix is specified, the file name will end with that suffix, otherwise there will be no
                        suffix.
        prefix (str):  If prefix is specified, the file name will begin with that prefix; otherwise,
                        a default prefix is used.
        dir (str):  If dir is specified, the file will be created in that directory; otherwise, a default directory is
                        used.
    Returns:
        str: path to the directory
    """
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    yield tmp
    shutil.rmtree(tmp)


if __name__ == '__main__':
    cli()
