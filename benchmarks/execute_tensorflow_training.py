#!/usr/bin/env python

from __future__ import absolute_import

import contextlib
import itertools
import json
import os
import shutil
import subprocess
import tempfile

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sagemaker import Session
from sagemaker.tensorflow import TensorFlow

dir_path = os.path.dirname(os.path.realpath(__file__))
benchmark_results_dir = os.path.join('s3://', Session().default_bucket(), 'tf-benchmarking')


@click.group()
def cli():
    pass


def generate_report():
    results_dir = os.path.join(dir_path, 'results')

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    subprocess.call(['aws', 's3', 'cp', '--recursive', benchmark_results_dir, results_dir])

    jobs = {}

    for job_name in os.listdir(results_dir):
        jobs[job_name] = {}

        _, instance_type, instance_count, device, py_version, _, _, _, _, _, _, _, _, = job_name.split('-')

        current_dir = os.path.join(results_dir, job_name)

        model_dir = os.path.join(current_dir, 'output', 'model.tar.gz')
        subprocess.call(['tar', '-xvzf', model_dir], cwd=current_dir)

        jobs[job_name]['instance_type'] = instance_type
        jobs[job_name]['instance_count'] = instance_count
        jobs[job_name]['device'] = device
        jobs[job_name]['py_version'] = py_version

        benchmark_log = os.path.join(current_dir, 'benchmark_run.log')

        if os.path.exists(benchmark_log):
            with open(benchmark_log) as f:
                data = json.load(f)


                jobs[job_name]['dataset'] = data['dataset']['name']
                jobs[job_name]['num_cores'] = data['machine_config']['cpu_info']['num_cores']
                jobs[job_name]['cpu_info'] = data['machine_config']['cpu_info']['cpu_info']
                jobs[job_name]['mhz_per_cpu'] = data['machine_config']['cpu_info']['mhz_per_cpu']
                jobs[job_name]['gpu_count'] = data['machine_config']['gpu_info']['count']
                jobs[job_name]['gpu_model'] = data['machine_config']['gpu_info']['model']

                def find_value(parameter):
                    other_key = [k for k in parameter if k != 'name'][0]
                    return parameter[other_key]

                for parameter in data['run_parameters']:
                    jobs[job_name][parameter['name']] = find_value(parameter)

                jobs[job_name]['model_name'] = data['model_name']
                jobs[job_name]['run_date'] = data['run_date']
                jobs[job_name]['tensorflow_version'] = data['tensorflow_version']['version']
                jobs[job_name]['tensorflow_version_git_hash'] = data['tensorflow_version']['git_hash']

        with open(os.path.join(current_dir, 'metric.log')) as f:
            for line in f.readlines():
                metric = json.loads(line)
                metric_name = metric['name']
                metric_value = metric['value']

                current_value = jobs[job_name].get(metric_name)
                if current_value and isinstance(current_value, list):
                    jobs[job_name][metric_name].append(metric_value)
                elif current_value:
                    jobs[job_name][metric_name] = [current_value, metric_value]
                else:
                    jobs[job_name][metric_name] = metric_value

    df = pd.DataFrame(jobs)

    grouped_by_instance = df[df.T.groupby(['batch_size']).groups[256]].T.groupby('instance_type')
    [df[grouped_by_instance.groups[x]].T['average_examples_per_sec'].mean() for x, i in grouped_by_instance], [x for x, i in grouped_by_instance]

    batch_size_256 = [df[grouped_by_instance.groups[x]].T['average_examples_per_sec'].mean() for x, i in grouped_by_instance]

    grouped_by_instance = df[df.T.groupby(['batch_size']).groups[512]].T.groupby('instance_type')
    batch_size_512 = [df[grouped_by_instance.groups[x]].T['average_examples_per_sec'].mean() for x, i in grouped_by_instance]

    plot_comparison([x for x, i in grouped_by_instance], batch_size_256, batch_size_512, 'batch size 256', 'batch size 523')
    return df


def plot_comparison(x_labels, left_group, right_group, left_label, right_label):
    ind = np.arange(len(left_group))  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width / 2, left_group, width, color='SkyBlue', label=left_label)
    rects2 = ax.bar(ind + width / 2, right_group, width, color='IndianRed', label=right_label)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Examples per second')
    ax.set_title('Average Examples per second per instance type')
    ax.set_xticks(ind)
    ax.set_xticklabels(x_labels)
    ax.legend()

    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                    '{}'.format(int(height)), ha=ha[xpos], va='bottom')

    autolabel(rects1, "left")
    autolabel(rects2, "right")
    plt.show()


def _plot_column(column, grouped_by_nodes):
    _, ax = plt.subplots()
    for label, df in grouped_by_nodes:
        df[column].plot(ax=ax, label=label, title='%s per training job' % column)
    plt.legend()


@cli.command('train')
@click.option('--framework-version', required=True, type=click.Choice(['1.11', '1.12']))
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

        PYTHONPATH="/opt/ml/code/models:$PYTHONPATH" python benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py %s \
        --train_dir /opt/ml/model --eval_dir /opt/ml/model --batch_size %s\n"""

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
                dependencies=[os.path.join(dir_path, 'benchmarks'),
                              os.path.join(dir_path, 'models')],
                base_job_name=base_name,
                train_instance_count=instance_count,
                train_instance_type=instance_type,
                framework_version=framework_version,
                py_version=py_version,
                script_mode=True,
                hyperparameters={
                    'sagemaker_mpi_enabled': True,
                    'sagemaker_mpi_num_of_processes_per_host': 8,
                    'sagemaker_mpi_custom_mpi_options': '-x HOROVOD_TIMELINE --output-filename /opt/ml/model/hlog'
                },
                output_path=benchmark_results_dir,
                security_group_ids=security_groups,
                subnets=subnets
            )

            estimator.fit(wait=wait)

            if wait:
                artifacts_path = os.path.join(dir_path, 'results',
                                              estimator.latest_training_job.job_name)
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

    return 'tf-%s-%s-%s-%s-b%s' % (
        instance_typename, instance_count, device, python_version, batch_size)


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
