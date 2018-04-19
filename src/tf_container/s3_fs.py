import os

import boto3
from botocore.exceptions import ClientError

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
    os.environ['S3_USE_HTTPS'] = '1'


def get_default_bucket(region):
    account = boto3.client('sts').get_caller_identity()['Account']
    default_bucket = 'sagemaker-{}-{}'.format(region, account)

    s3 = boto3.client('s3', region_name=region)
    try:
        # 'us-east-1' cannot be specified because it is the default region:
        # https://github.com/boto/boto3/issues/125
        if region == 'us-east-1':
            s3.create_bucket(Bucket=default_bucket)
        else:
            s3.create_bucket(Bucket=default_bucket, CreateBucketConfiguration={'LocationConstraint': region})

        _logger.info('Created S3 bucket: {}'.format(default_bucket))
    except ClientError as e:
        error_code = e.response['Error']['Code']
        message = e.response['Error']['Message']

        if error_code == 'BucketAlreadyOwnedByYou':
            pass
        elif error_code == 'OperationAborted' and 'conflicting conditional operation' in message:
            # If this bucket is already being concurrently created, we don't need to create it again.
            pass
        elif error_code == 'TooManyBuckets':
            # Succeed if the default bucket exists
            try:
                s3.meta.client.head_bucket(Bucket=default_bucket)
                pass
            except ClientError:
                raise
        else:
            raise

    return 's3://{}'.format(default_bucket)
