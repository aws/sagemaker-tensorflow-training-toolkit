# TensorFlow benchmarking scripts

This folder contains the TF training scripts https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks.

## Basic usage
**execute_tensorflow_training.py** uses SageMaker python sdk to start a training job. It takes the following parameters:

- role: SageMaker role used for training
- region: SageMaker region
- py-versions: py2 or py3 or "py2, py3"
- instance-types: A list of SageMaker instance types, for example 'ml.p2.xlarge, ml.c4.xlarge'. Use 'local' for local mode training.
- checkpoint-path: The S3 location where the model checkpoints and tensorboard events are saved after training 

Any unknown arguments will be passed to the training script as additional arguments.

## Examples:

```bash
./execute_tensorflow_training.py -t local -r SageMakerRole --instance-type local  --num_epochs 1 --wait

./execute_tensorflow_training.py -t local -r SageMakerRole --instance-type ml.c4.xlarge, ml.c5.xlarge  --model resnet50

```

## Using other models, datasets and benchmarks configurations
```python tf_cnn_benchmarks/tf_cnn_benchmarks.py --help``` shows all the options that the script has.


## Tensorboard events and checkpoints

Tensorboard events are being saved to the S3 location defined by the hyperparameter checkpoint_path during training. That location can be overwritten by setting the script argument ```checkpoint-path```:

```bash
python execute_tensorflow_training.py ... --checkpoint-path s3://my/bucket/output/data
```
