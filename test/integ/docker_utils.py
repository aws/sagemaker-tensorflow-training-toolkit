#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

from __future__ import absolute_import

import logging
import os
import subprocess
import sys
import tempfile
import uuid
from time import sleep

logger = logging.getLogger(__name__)

CYAN_COLOR = '\033[36m'
END_COLOR = '\033[0m'


def registry(aws_id, region):
    return '{}.dkr.ecr.{}.amazonaws.com'.format(aws_id, region)


def train(image_name, resource_folder, processor):
    docker = 'docker' if processor == 'cpu' else 'nvidia-docker'

    cmd = [docker,
           'run',
           '--rm',
           '-h', 'algo-1',
           '-v', '{}:/opt/ml'.format(resource_folder),
           '-e', 'AWS_ACCESS_KEY_ID',
           '-e', 'AWS_SECRET_ACCESS_KEY',
           '-e', 'AWS_SESSION_TOKEN',
           image_name, 'train']
    check_call(cmd)


def check_call(cmd, *popenargs, **kwargs):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")
    _print_cmd(cmd)
    subprocess.check_call(cmd, *popenargs, **kwargs)


def _print_cmd(cmd):
    print('executing docker command: {}{}{}'.format(CYAN_COLOR, ' '.join(cmd), END_COLOR))
    sys.stdout.flush()


class Container(object):
    def __init__(self, image, processor, startup_delay=1):
        self.temp_dir = tempfile.gettempdir()
        self.image = image
        self.name = str(uuid.uuid4())
        self.startup_delay = startup_delay
        self.docker = 'docker' if processor == 'cpu' else 'nvidia-docker'

    def __enter__(self):
        print('in container.enter for container ' + self.image + ',' + self.name)
        self.remove_container()

        cmd = [self.docker,
               'run',
               '-d',
               '-t',
               '-e', 'AWS_ACCESS_KEY_ID',
               '-e', 'AWS_SECRET_ACCESS_KEY',
               '-e', 'AWS_SESSION_TOKEN',
               '--entrypoint', 'bash',
               '--name', self.name,
               self.image]

        check_call(cmd)

        # waiting for the server to spin up
        sleep(self.startup_delay)

        self.execute_command(['pip', 'install', 'requests'])
        self.execute_command(['pip', 'install', 'pytest'])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_container()

    def remove_container(self):
        cmd = [self.docker,
               'rm',
               '-f',
               self.name]

        try:
            check_call(cmd)
        except:
            pass

    def copy(self, src, dst):
        cmd = [self.docker,
               'cp',
               src,
               '{}:{}'.format(self.name, dst)]

        check_call(cmd)

    def execute_command(self, cmd):

        docker_cmd = [self.docker, 'exec', '-t', self.name]
        docker_cmd.extend(cmd)

        _print_cmd(docker_cmd)

        lines = []
        process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE)
        print(
        '{}============================= container output ============================='.format(
            CYAN_COLOR))
        for line in iter(process.stdout.readline, b''):
            sys.stdout.write(line.decode('utf-8'))
            sys.stdout.flush()
            lines.append(line.decode('utf-8'))
        msg = '\n{}============================= end of container output ============================='
        print(msg.format(CYAN_COLOR))

        process.wait()

        warnings = 0
        for line in lines:
            if line.startswith('WARNING'):
                warnings += 1
                print(line)
            else:
                break
        output = '\n'.join(lines[warnings:])

        if process.returncode != 0:
            print("docker exec error. output:\n{}".format(output))
            raise ValueError("non-zero exit code: {}".format(process.returncode))

        return output

    def execute_pytest(self, tests_path):
        container_test_path = '/root/{}'.format(os.path.basename(tests_path))
        self.copy(tests_path, container_test_path)
        return self.execute_command(['pytest', '-vv', '-s', '--color=yes', container_test_path])


class HostingContainer(Container):
    def __init__(self, image, opt_ml, script_name, processor, requirements_file=None,
                 startup_delay=5, region=None):
        super(HostingContainer, self).__init__(image=image,
                                               processor=processor,
                                               startup_delay=startup_delay)
        self.opt_ml = opt_ml
        self.script_name = script_name
        self.opt_ml = opt_ml
        self.requirements_file = requirements_file
        self.region = region

    def __enter__(self):
        cmd = [self.docker,
               'run',
               '-d',
               '-h', 'algo-1',
               '-v', '{}:/opt/ml'.format(self.opt_ml),
               '-e', 'AWS_ACCESS_KEY_ID',
               '-e', 'AWS_SECRET_ACCESS_KEY',
               '-e', 'AWS_SESSION_TOKEN',
               '-e', 'SAGEMAKER_REGION={}'.format(self.region if self.region else ''),
               '-e', 'SAGEMAKER_CONTAINER_LOG_LEVEL=20',
               '-e', 'SAGEMAKER_PROGRAM={}'.format(self.script_name),
               '-e', 'SAGEMAKER_REQUIREMENTS={}'.format(self.requirements_file),
               '--name', self.name,
               self.image, 'serve']

        check_call(cmd)

        # waiting for the server to spin up
        sleep(self.startup_delay)

        self.execute_command(['pip', 'install', 'requests'])
        self.execute_command(['pip', 'install', 'pytest'])

        return self

    def invoke_endpoint(self, input, content_type='application/json', accept='application/json'):
        return self.execute_command([
            'curl',
            '-f',
            '-H', 'Content-Type: {}'.format(content_type),
            '-H', 'Accept: {}'.format(accept),
            '-d', str(input),
            'http://127.0.0.1:8080/invocations'
        ])

    def ping(self):
        self.execute_command(['curl', '-f', '-v', 'http://localhost:8080/ping'])
