import os
import boto3
import container_support as cs


def configure_s3_fs(checkpoint_path):
    # If env variable is not set, defaults to None, which will use the global endpoint.
    region_name = os.environ.get('AWS_REGION')
    s3 = boto3.client('s3', region_name=region_name)

    # We get the AWS region of the checkpoint bucket, which may be different from
    # the region this container is currently running in.
    bucket_name, key = cs.parse_s3_url(checkpoint_path)
    bucket_location = s3.get_bucket_location(Bucket=bucket_name)['LocationConstraint']

    # Configure environment variables used by TensorFlow S3 file system
    if bucket_location:
        os.environ['S3_REGION'] = bucket_location

    # setting log level to WARNING
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['S3_USE_HTTPS'] = '1'
