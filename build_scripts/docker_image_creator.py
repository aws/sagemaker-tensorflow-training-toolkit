""" Script to create Sagemaker TensorFlow Docker images

    Usage:
        python docker_image_creator.py optimized_binary_link gpu|cpu tensorflow_version python_version
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

def create_docker_image(processor, framework_version, python_version, optbin_path):
    """ Function builds a docker image

    Args:
        processor (str): gpu or cpu
        framework_version (str): tensorflow version i.e 1.6.0
        python_version (str): (i.e. 3.6.5 or 2.7.4)
        optbin_path (str): link to where the optimized binary is.
    """
    # Initialize commonly used variables
    py_v = 'py{}'.format(python_version.split('.')[0]) # i.e. py2

    # If necessary, get optimized binary and put in final docker image repo
    if optbin_path:
        print('Getting optimized binary...')
        # Check if it is local or remote
        optbin_filename = 'tensorflow-{}-{}-binary.whl'.format(framework_version, processor)
        if os.path.isfile(optbin_path):
            shutil.copyfile(optbin_filename, '{}/../docker/{}/final/{}/{}'.format(PATH_TO_SCRIPT, framework_version, py_v, output_filename))
        else:
            with open('{}/../docker/{}/final/{}/{}'.format(PATH_TO_SCRIPT, framework_version, py_v, optbin_filename), 'wb') as optbin_file:
                subprocess.call(['curl', optbin_link], stdout=optbin_file)

    # Build base image
    print('Building base image...')
    image_name = 'tensorflow-base:{}-{}-{}'.format(framework_version, processor,  py_v)
    base_docker_path = '{}/../docker/{}/base/Dockerfile.{}'.format(PATH_TO_SCRIPT, framework_version, processor)
    subprocess.call([DOCKER, 'build', '-t', image_name, '-f', base_docker_path, '.'])

    #  Build final image
    print('Building final image...')
    subprocess.call(['python', 'setup.py', 'sdist'], cwd='{}/..'.format(PATH_TO_SCRIPT))
    output_file = glob.glob('{}/../dist/sagemaker_tensorflow_container-*.tar.gz'.format(PATH_TO_SCRIPT))[0]
    output_filename = output_file.split('/')[-1]
    shutil.copyfile(output_file, '{}/../docker/{}/final/{}/{}'.format(PATH_TO_SCRIPT, framework_version, py_v, output_filename))
    if optbin_path:
        subprocess.call([DOCKER, 'build', '-t', 'preprod-tensorflow:{}-{}-{}'.format(framework_version, processor, py_v),
                        '--build-arg', 'py_version={}'.format(py_v[-1]), '--build-arg', 'framework_installable={}'.format(optbin_filename),
                        '-f', 'Dockerfile.{}'.format(processor), '.'], cwd='{}/../docker/{}/final/{}'.format(PATH_TO_SCRIPT, framework_version, py_v))
    else:
        subprocess.call([DOCKER, 'build', '-t', 'preprod-tensorflow:{}-{}-{}'.format(framework_version, processor, py_v),
                        '--build-arg', 'py_version={}'.format(py_v[-1]), '-f', 'Dockerfile.{}'.format(processor), '.'],
                        cwd='{}/../docker/{}/final/{}'.format(PATH_TO_SCRIPT, framework_version, py_v))

if __name__ == '__main__':
    # Parse command line options
    parser = argparse.ArgumentParser(description='Build Sagemaker TensorFlow Docker Images')
    parser.add_argument('processor_type', choices=['cpu', 'gpu'], help='gpu if you would like to use GPUs or cpu')
    parser.add_argument('framework_version', help='TensorFlow framework version (i.e. 1.8.0)')
    parser.add_argument('python_version', help='Python version to be used (i.e. 2.7.0)')
    parser.add_argument('--optimized_binary_path', default=None, help='path to optimized binary')
    parser.add_argument('--nvidia-docker', action='store_true', help="Enables nvidia-docker usage over docker usage")
    args = parser.parse_args()

    # Set value for docker
    DOCKER = 'nvidia-docker' if args.nvidia_docker else 'docker'

    # Sets PATH_TO_SCRIPT so that command can be run from anywhere
    PATH_TO_SCRIPT = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Build image
    create_docker_image(args.processor_type, args.framework_version, args.python_version, args.optimized_binary_path)
