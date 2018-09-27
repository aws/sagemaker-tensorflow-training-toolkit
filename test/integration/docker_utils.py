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
import subprocess
import sys
import tempfile
import uuid

logger = logging.getLogger(__name__)

CYAN_COLOR = '\033[36m'
END_COLOR = '\033[0m'


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
               '--entrypoint', 'bash',
               '--name', self.name,
               self.image]

        check_call(cmd)

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
