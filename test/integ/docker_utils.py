from __future__ import absolute_import

import base64
import logging
import subprocess
import sys
import tempfile

import boto3
import fasteners

logger = logging.getLogger(__name__)

DOCKER_API_VERSION = '1.24'
DOCKER_API_URL = 'unix:///var/run/docker.sock'
DOCKER_LOCKFILE = '{}/docker.lock'.format(tempfile.gettempdir())

CYAN_COLOR = '\033[36m'
END_COLOR = '\033[0m'


def registry(aws_id, region):
    return '{}.dkr.ecr.{}.amazonaws.com'.format(aws_id, region)


# interprocess lock so concurrent tests don't mangle the docker config.json file
@fasteners.interprocess_locked(DOCKER_LOCKFILE)
def login(aws_id, region):
    ecr_client = boto3.Session(region_name=region).client('ecr')
    response = ecr_client.get_authorization_token()

    token = response['authorizationData'][0]['authorizationToken']
    auth_token = base64.b64decode(token).decode()
    password = auth_token.split(':')[1]

    check_call(
        ['docker', 'login', '-u', 'AWS', '-p', password, 'https://{}'.format(registry(aws_id, region))])


def check_call(cmd, *popenargs, **kwargs):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")
    _print_cmd(cmd)
    subprocess.check_call(cmd, *popenargs, **kwargs)


def _print_cmd(cmd):
    print('executing docker command: {}{}{}'.format(CYAN_COLOR, ' '.join(cmd), END_COLOR))
    sys.stdout.flush()
