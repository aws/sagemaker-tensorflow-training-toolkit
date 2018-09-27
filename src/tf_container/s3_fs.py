import os
import boto3
import container_support as cs


def configure_s3_fs(checkpoint_path, region_name=None):
    s3 = boto3.client('s3', region_name=region_name)

    # We get the AWS region of the checkpoint bucket, which may be different from
    # the region this container is currently running in.
    bucket_name, key = cs.parse_s3_url(checkpoint_path)
    bucket_location = s3.get_bucket_location(Bucket=bucket_name)['LocationConstraint']

    # Configure environment variables used by TensorFlow S3 file system
    if bucket_location:
        os.environ['S3_REGION'] = bucket_location
    os.environ['S3_USE_HTTPS'] = '1'
