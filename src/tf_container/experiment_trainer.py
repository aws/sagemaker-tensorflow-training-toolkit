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

import boto3
import inspect
import os
import tensorflow as tf
import tf_container.s3_fs as s3_fs
from tf_container.run import logger
from tensorflow.contrib.learn import RunConfig, Experiment
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.contrib.training import HParams


class Trainer(object):
    DEFAULT_TRAINING_CHANNEL = 'training'

    def __init__(self,
                 customer_script,
                 current_host,
                 hosts,
                 train_steps=1000,
                 eval_steps=100,
                 input_channels=None,
                 model_path=None,
                 output_path=None,
                 min_eval_frequency=1000,
                 customer_params={},
                 save_checkpoints_secs=300):
        """

        Args:
            customer_script: (module) Customer loaded module
            current_host: (str) Current hostname
            hosts: list (str) List with all containers list names
            train_steps: (int) Perform this many steps of training. 'None', the default,
                means train forever.
            eval_steps: (int) 'evaluate' runs until input is exhausted (or another exception
                is raised), or for 'eval_steps' steps, if specified.
            input_channels: (dict) Dictionary with input channels
            model_path: (str) Directory where checkpoints will be saved. Can be a S3 bucket
            output_path: (str) Local directory where the model will be saved
            min_eval_frequency: (int) Applies only to master container. the minimum
                number of steps between evaluations. Of course, evaluation does not
                occur if no new snapshot is available, hence, this is the minimum.
                If 0, the evaluation will only happen after training.
                Defaults to 1000.
        """
        self.customer_script = customer_script
        self.current_host = current_host
        self.hosts = hosts
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.input_channels = input_channels
        self.model_path = model_path
        self.ouput_path = output_path
        self.task_type = None

        customer_params['min_eval_frequency'] = customer_params.get('min_eval_frequency', min_eval_frequency)
        customer_params['save_checkpoints_secs'] = customer_params.get('save_checkpoints_secs', save_checkpoints_secs)

        self.customer_params = customer_params

        if model_path.startswith('s3://'):
            s3_fs.configure_s3_fs(model_path)

    def _get_task_type(self, masters):
        if self.current_host in masters:
            return 'master'
        return 'worker'

    def build_tf_config(self):
        """Builds a dictionary containing cluster information based on number of hosts and number of parameter servers.
        More information about TF_Config: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn
        /python/learn/estimators/run_config.py#L77
        :return: task_type and tf_config dictionary
        """

        masters = self.hosts[:1]
        workers = self.hosts[1:]
        ps = self.hosts[:] if len(self.hosts) > 1 else None

        self.task_type = self._get_task_type(masters)

        task_map = {'master': masters, 'worker': workers}

        if ps:
            task_map['ps'] = ps

        task_id = task_map[self.task_type].index(self.current_host)

        def build_host_addresses(my_hosts, port='2222'):
            return ['{}:{}'.format(host, port) for host in my_hosts]

        tf_config = {
            "cluster": {
                "master": build_host_addresses(masters),
            },
            "task": {
                "index": task_id,
                "type": self.task_type
            },
            "environment": 'cloud'
        }

        if ps:
            tf_config['cluster']['ps'] = build_host_addresses(ps, port='2223')

        if len(workers) > 0:
            tf_config['cluster']['worker'] = build_host_addresses(workers)

        return tf_config

    def train(self):
        experiment_fn = self._generate_experiment_fn()
        hparams = HParams(**self.customer_params)

        learn_runner.run(experiment_fn,
                         run_config=self._build_run_config(),
                         hparams=hparams)

    def saves_training(self):
        return hasattr(self.customer_script, "serving_input_fn")

    def _resolve_value_for_training_input_fn_parameter(self, alias_key):
        """
        Handle potentially aliased key name and return value for valid one.

        :return: value for the requested parameter or None
        """
        key_mappings = {('training_dir', 'dir'): 'training_dir',
                        ('hyperparameters', 'params'): 'hyperparameters',
                        ('input_channels', 'channels'): 'input_channels'}
        resolved_key = None
        for k, v in key_mappings.items():
            if alias_key in k:
                resolved_key = v
                break

        parameter_values = {'training_dir': self.input_channels.get(self.DEFAULT_TRAINING_CHANNEL, None),
                            'hyperparameters': self.customer_params,
                            'input_channels': self.input_channels}
        return parameter_values[resolved_key] if resolved_key else None

    def _generate_experiment_fn(self):
        def _experiment_fn(run_config, hparams):
            valid_experiment_keys = ['eval_metrics', 'train_monitors', 'eval_hooks', 'local_eval_frequency',
                                     'eval_delay_secs', 'continuous_eval_throttle_secs', 'min_eval_frequency',
                                     'delay_workers_by_global_step', 'train_steps_per_iteration']

            experiment_params = {k: v for k, v in self.customer_params.items() if k in valid_experiment_keys}

            logger.info("creating Experiment:")
            logger.info(experiment_params)

            '''
            TensorFlow input functions (train_input_fn, and eval_input_fn) can return features and
            labels, or a function that returns features and labels
            Examples of valid input functions:

                def train_input_fn(training_dir, hyperparameters):
                    ...
                    return tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels)

                def train_input_fn(training_dir, hyperparameters):
                    ...
                    return features, labels
            '''
            def _train_input_fn():
                """Prepare parameters for the train_input_fn and invoke it"""
                declared_args = inspect.getargspec(self.customer_script.train_input_fn)
                invoke_args = {arg: self._resolve_value_for_training_input_fn_parameter(arg)
                               for arg in declared_args.args}
                return _function(self.customer_script.train_input_fn(**invoke_args))()

            def _eval_input_fn():
                declared_args = inspect.getargspec(self.customer_script.eval_input_fn)
                invoke_args = {arg: self._resolve_value_for_training_input_fn_parameter(arg)
                               for arg in declared_args.args}
                return _function(self.customer_script.eval_input_fn(**invoke_args))()

            '''
            TensorFlow serving input functions (serving_input_fn) can return a ServingInputReceiver object or a
            function that a ServingInputReceiver
            Examples of valid serving input functions:

                def serving_input_fn(params):
                    feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[4])}
                    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

                def serving_input_fn(hyperpameters):
                    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, 32, 32, 3])}
                    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
            '''
            def _serving_input_fn():
                return _function(self.customer_script.serving_input_fn(self.customer_params))()

            def _export_strategy():
                if self.saves_training():
                    return [saved_model_export_utils.make_export_strategy(
                        serving_input_fn=_serving_input_fn,
                        default_output_alternative_key=None,
                        exports_to_keep=1)]
                logger.warn("serving_input_fn not specified, model NOT saved, use checkpoints to reconstruct")
                return None

            return Experiment(
                estimator=self._build_estimator(run_config=run_config, hparams=hparams),
                train_input_fn=_train_input_fn,
                eval_input_fn=_eval_input_fn,
                export_strategies=_export_strategy(),
                train_steps=self.train_steps,
                eval_steps=self.eval_steps,
                **experiment_params
            )

        return _experiment_fn

    def _build_run_config(self):
        valid_runconfig_keys = ['save_summary_steps', 'save_checkpoints_secs', 'save_checkpoints_steps',
                                'keep_checkpoint_max', 'keep_checkpoint_every_n_hours', 'log_step_count_steps']

        runconfig_params = {k: v for k, v in self.customer_params.items() if k in valid_runconfig_keys}

        logger.info("creating RunConfig:")
        logger.info(runconfig_params)

        run_config = RunConfig(
            model_dir=self.model_path,
            **runconfig_params
        )
        return run_config

    def _build_estimator(self, run_config, hparams):
        # hparams is of type HParams at this point but all the interface functions are assuming dict
        hyperparameters = hparams.values()

        if hasattr(self.customer_script, 'estimator_fn'):
            logger.info("invoking estimator_fn")
            return self.customer_script.estimator_fn(run_config, hyperparameters)
        elif hasattr(self.customer_script, 'keras_model_fn'):
            logger.info("involing keras_model_fn")
            model = self.customer_script.keras_model_fn(hyperparameters)
            return tf.keras.estimator.model_to_estimator(keras_model=model, config=run_config)
        else:
            logger.info("creating the estimator")

            def _model_fn(features, labels, mode, params):
                return self.customer_script.model_fn(features, labels, mode, params)

            return tf.estimator.Estimator(
                model_fn=_model_fn,
                params=hyperparameters,
                config=run_config)


def _function(object):
    """Ensures that the object is a function. Wraps the object in a function otherwise.
    Args:
        object: object to be wrapped as function

    Returns: function with the wrapped object.
    """
    if hasattr(object, '__call__'):
        return object

    return lambda: object
