""" Script to create Sagemaker TensorFlow Docker image
    Run the command:
        python docker_image_creator.py optimized_binary_link gpu|cpu tensorflow_version python_version

"""

import argparse
import glob
import subprocess
import sys

## GLOBALS
DOCKER = 'docker' # Stores which type of DOCKER to use, nvidia-docker or docker
PATH_TO_SCRIPT = ''  # Stores route the user takes to build scripts directory

def create_docker_image(optbin_link, processor, framework_version, python_version):
    """ Function builds a docker image with the TF optimized binary

    Args:
        param1 (str): optbin_link - link to where the optimized binary is
        param2 (str): processor - gpu or cpu
        param3 (str): framework_version - tensorflow version i.e 1.6.0
        param4 (str): python_version (i.e. 3.6.5 or 2.7.4)

    """
    # 1.) Initialize commonly used variables
    py_v = 'py{}'.format(python_version.split('.')[0]) # i.e. py2

    # 2.) Get optimized binary - and put in final docker image repo
    print('Getting optimized binary...')
    optbin_filename = 'tensorflow-{}-cp27-cp27mu-manylinux1_x86_64.whl'.format(framework_version)
    with open('{}/../docker/{}/final/{}/{}'.format(PATH_TO_SCRIPT, framework_version, py_v, optbin_filename), 'wb') as optbin_file:
        subprocess.call(['curl', optbin_link], stdout=optbin_file)

    # 3.) Build base image
    print('Building base image...')
    image_name = 'tensorflow-base:{}-{}-{}'.format(framework_version, processor,  py_v)
    base_docker_path = '{}/../docker/{}/base/Dockerfile.{}'.format(PATH_TO_SCRIPT, framework_version, processor)
    subprocess.call(['sudo', DOCKER, 'build', '-t', image_name, '-f', base_docker_path, '.'])

    # 4.) Build final image
    print('Building final image...')
    subprocess.call(['python', 'setup.py', 'sdist'], cwd='{}/..'.format(PATH_TO_SCRIPT))
    output_file = glob.glob('{}/../dist/sagemaker_tensorflow_container-*.tar.gz'.format(PATH_TO_SCRIPT))[0] # use glob to use regex
    subprocess.call(['cp', output_file, '{}/../docker/{}/final/{}'.format(PATH_TO_SCRIPT, framework_version, py_v)])
    subprocess.call(['sudo', DOCKER, 'build', '-t', 'preprod-tensorflow:{}-{}-{}'.format(framework_version, processor, py_v), \
                    '--build-arg', 'py_version={}'.format(py_v[-1]), '--build-arg', 'framework_installable={}'.format(optbin_filename), \
                    '-f', 'Dockerfile.{}'.format(processor), '.'], cwd='{}/../docker/{}/final/{}'.format(PATH_TO_SCRIPT, framework_version, py_v))

if __name__ == '__main__':
    # Parse command line options
    parser = argparse.ArgumentParser(description='Build Sagemaker TensorFlow Docker Images')
    parser.add_argument('optimized_binary_link', help='link to place with optimized binary')
    parser.add_argument('processor_type', choices=['cpu', 'gpu'], help='gpu if you would like to use GPUs or cpu')
    parser.add_argument('framework_version', help='TensorFlow framework version (i.e. 1.8.0)')
    parser.add_argument('python_version', help='Python version to be used (i.e. 2.7.0)')
    parser.add_argument('--nvidia-docker', action='store_true', help="Enables nvidia-docker usage over docker usage")
    args = parser.parse_args()

    # Set value for docker
    DOCKER = 'nvidia-docker' if args.nvidia_docker else DOCKER

    # Sets PATH_TO_SCRIPT so that command can be run from anywhere
    PATH_TO_SCRIPT = '/'.join(sys.argv[0].split('/')[:-1])
    PATH_TO_SCRIPT = '.' if PATH_TO_SCRIPT == '' else PATH_TO_SCRIPT

    # Build image
    create_docker_image(args.optimized_binary_link, args.processor_type, args.framework_version, args.python_version)
