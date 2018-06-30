"""
Script to run docker image creation
Run the command:
    python docker_image_creator.py optimized_binary_link gpu|cpu tensorflow_version python_version
"""
import argparse
import os
import sys

## GLOBALS
# Allows user to run script anywhere
BASE_PATH = ""

def create_docker_image(optbin_link, processor, framework_version, python_version):
    """
    Function builds a docker image with the TF optimized binary

    Assumptions:
        1. Script needs to be run inside of the build_scripts files

    :param optbin_link: link to where the optimized binary is
    :param processor: gpu or cpu
    :param framework_version: tensorflow version i.e 1.6.0
    :param python_version: version of python to build container with i.e. 3.6.5 or 2.7.4
    """
    # 1.) Initialize some commonly used variables
    pyV = "py{}".format(python_version.split('.')[0]) # i.e. py2
    framework = "tensorflow"
    # 2.) Get optimized binary - and put in final docker image repo
    print("Getting optimized binary...")
    optbin_filename = "{}-{}-cp27-cp27mu-manylinux1_x86_64.whl".format(framework, framework_version)
    os.system("curl {} > {}/../docker/{}/final/{}/{}".format(BASE_PATH, optbin_link, framework_version, pyV, optbin_filename)) # ADDED BASE_PATH
    # 3.) Build base image
    print("Building base image...")
    image_name = "{}-base:{}-{}-{}".format(framework, framework_version, processor,  pyV)
    base_docker_path = "{}/../docker/{}/base/Dockerfile.{}".format(framework_version, processor)  # ADDED BASE_PATH
    os.system("sudo nvidia-docker build -t {} -f {} .".format(image_name, base_docker_path))
    # 4.) Build final image
    print("Building final image...")
    os.chdir("{}/..".format(BASE_PATH)) # ADDED BASE_PATH
    os.system("python setup.py sdist")
    os.system("cp dist/sagemaker_tensorflow_container-*.tar.gz docker/{}/final/{}".format(framework_version, pyV))
    os.chdir("docker/{}/final/{}".format(framework_version, pyV))
    os.system \
        ("sudo nvidia-docker build -t preprod-{}:{}-{}-{} --build-arg py_version={} --build-arg framework_installable={}  -f Dockerfile.{} .".format
            (framework, framework_version, processor, pyV, pyV[-1], optbin_filename, processor))
    # 5.) Return to build_scripts directory
    os.chdir("{}".format(BASE_PATH)) # ADDED BASE_PATH

if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Build Sagemaker Tensorflow Docker Images")
    parser.add_argument("optimized_binary_link", help="link to place with optimized binary")
    parser.add_argument("processor_type", help="gpu if you would like to use GPUs or cpu")
    parser.add_argument("framework_version", help="Tensorflow framework version (i.e. 1.8.0)")
    parser.add_argument("python_version", help="Python version to be used (i.e. 2.7.0)")
    args = parser.parse_args()
    # Sets BASE_PATH so that command can be run from anywhere
    BASE_PATH = "/".join(sys.argv[0].split('/')[:-1])
    BASE_PATH = "." if BASE_PATH == "" else BASE_PATH
    # Build image
    create_docker_image(args.optimized_binary_link, args.processor_type, args.framework_version, args.python_version)