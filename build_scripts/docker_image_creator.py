"""
Implements the parent function for docker image creation during single instance configuration
"""
import argparse
import os

def tensorflow_create_docker(dockerfile_github_link, optbin_link, gpu, framework_version, python_version):
    """
    Function builds a docker image with the TF optimized binary

    :param dockerfile_github_link: link to docker file repo
    :param optbin_link: link to where the optimized binary is
    :param gpu: True for gpu false for cpu
    :param framework_version: deep learning framework version i.e 1.6.0
    :param python_version: version of python to build container with i.e. 3.6.5 or 2.7.4
    :return: final built imageid
    """
    # 0.) Initialize some commonly used variables
    processor = "gpu" if gpu else "cpu"
    dockerfile_repo_name = dockerfile_github_link.split('/')[-1].split('.')[0]
    pyV = "py{}".format(python_version.split('.')[0]) # i.e. py2
    framework = "tensorflow"
    # 1.) Clone dockerfile repo from Github
    os.system("git clone {}".format(dockerfile_github_link))
    # 2.) Get optimized binary - probably need to better parametrize the name
    optbin_filename = "{}-{}-cp27-cp27mu-manylinux1_x86_64.whl".format(framework, framework_version) # just a tmp file name
    os.system("curl {} > {}".format(optbin_link, optbin_filename))
    # 3.) Build base image
    image_name = "{}-base:{}-{}-{}".format(framework, framework_version, processor,  pyV)
    base_docker_path = "docker/{}/base/Dockerfile.{}".format(framework_version, processor)
    os.system("sudo nvidia-docker build -t {} -f {}/{} .".format(image_name, dockerfile_repo_name, base_docker_path))
    # 4.) Build final image
    os.chdir("{}".format(dockerfile_repo_name))
    os.system("python setup.py sdist")
    os.system("cp dist/sagemaker_tensorflow_container-1.0.0.tar.gz docker/{}/final/{}".format(framework_version, pyV))
    os.system("cp ../{}  docker/{}/final/{}".format(optbin_filename, framework_version, pyV))
    os.chdir("docker/{}/final/{}".format(framework_version, pyV))
    os.system \
        ("sudo nvidia-docker build -t preprod-{}:{}-{}-{} --build-arg py_version={} --build-arg framework_installable={}  -f Dockerfile.{} .".format
            (framework, framework_version, processor, pyV, pyV[-1], optbin_filename, processor))
    # 5.) Clean everything up
    print("Cleaning up...")
    os.chdir("../../../../..")
    os.system("sudo rm -r {} {}".format(optbin_filename, dockerfile_repo_name))
    # 6.) Return image id
    image_id = os.popen("sudo nvidia-docker images preprod-{}:{}-{}-{} -q".format(framework, framework_version, processor, python_version)).read()
    return image_id[:-1] # removes endline


if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("docker_file_github_link", help="link to github containing docker files")
    parser.add_argument("optimized_binary_link", help="link to place with optimized binary")
    parser.add_argument("processor_type", help="'gpu' if you would like to use GPUs or 'cpu'")
    parser.add_argument("framework_version", help="Tensorflow framework version (i.e. 1.8.0)")
    parser.add_argument("python_version", help="Python version to be used (i.e. 2.7.0)")
    args = parser.parse_args()
    # Build image
    tensorflow_create_docker(args.dockerfile_github_link, args.optimized_binary, args.gpu, args.framework_version, args.python_version)
