"""
This file contains random pieces of code that are necessary to make things run smoothly
"""

# Package imports
import configparser
import logging
import json
import ast
import sys
import re

# Aliased imports
import numpy as np

# Huggingface imports
from transformers import Trainer
from evaluate import load

# Torch imports
import torch
import torch.nn as nn


###################################################################################################


def text_to_list(text:str):
    """
    If you have a list-like or json object that was imported as a string, use this function to pass
    it to its proper type.

    Args:
        A sring containing a json-like object

    Output:
        A list decoded from the input
    """

    # Try to load the input as a json
    try:
        out = json.loads(text)

    # If that's not possible...
    except json.JSONDecodeError:

        # Interpret the text as its contents using the ast (as type) library
        try:
            out = ast.literal_eval(text)

        # If all else falis, rise an error
        except Exception as e:
            out = text
            logging.error("Could not pass '" + text + "' to list!")
            raise e
        
    return out


###################################################################################################


def generate_evaluator(metrics, external_metrics=None):
    """
    This function creates an instance of a function that evaluates the performance of a model.

    
    Args [generate_evaluator]

        - metrics           A list with the metrics that should be used for testing. The names of
                            the metrics must correspond with metrics available in Huggingface's
                            evaluate library. You could also pass just the name of a metric as a
                            string instead.
                            The format for each entry should be:
                                name:str when using a metric with no averaging
                                ( name:str , average_method:str ) when using averaging

        - external_entries  A list of metric objects that should also be used to evaluate the data.
                            An example of these could be those from sklearn.
                            NOTE: As of 2023-09-23 this hasn't been implemented yet.
                            Default: None


    Output [generate_evaluator]

        A function called compute_metrics. We give the arguments and metrics for it in the
        following paragraphs.


    Args [compute_metrics]

        - eval_pred     The predictions of the model as logits, either as a numpy array or a tensor

        - y_pred        The predictions of the model as one-hot vectors, either as a numpy array or
                        a tensor

        - y_real        The real values against which we are comparing, either as a numpy array or
                        a tensor


    Output [compute_metrics]

        A dictionary where the keys are the names and averaging methods passed and the values are
        the outputs of the respective metric. See those metrics' documentations to see what to
        expect. If the name was the only thing passed, the key will be name. If both a name and an
        averagingmethod were passed, the key will be name_averagingmethod.
    """

    # If only a string was passed, transform it into a list
    if isinstance(metrics, str):
        metrics = [metrics]

    # Initialize lists to store the data
    averages = []
    loaded_metrics = []
    metric_names = []

    # Save the info for each metric
    for metric in metrics:
        # When the entry is just the name
        if isinstance(metric, str):
            loaded_metrics.append(load(metric))
            averages.append(None)
            metric_names.append(metric)
        # When we are also given the average method
        else:
            loaded_metrics.append(load(metric[0]))
            averages.append(metric[1])
            metric_names.append(metric[0])


    # Define the compute_metrics function
    def compute_metrics(eval_pred=None, y_real=None, y_pred=None):

        # If logits are given, we turn them into predictions
        if eval_pred is not None:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
        # Else, we just keep the predictions
        elif (y_real is not None) and (y_pred is not None):
            predictions = y_pred
            labels = y_real
        # If no prediciton or logit was given, we raise an exception
        else:
            logging.critical("You must give either a ( logit , label ) pair or a set of y_real and y_predicted values!")
            raise Exception
        
        # Metric evaluation loop
        out = {}
        for i in range(len(metrics)):

            # Set the name of the metric
            name = metric_names[i]

            # Compute the values if no average was given
            if averages[i] is not None:
                name += "_" + averages[i]
                value = loaded_metrics[i].compute(references=labels,
                                                  predictions=predictions,
                                                  average=averages[i])
            
            # Compute the values if an average method was given
            else:
                value = loaded_metrics[i].compute(references=labels,
                                                  predictions=predictions)

            # Save the value of the metric
            value = value[list(value.keys())[0]]
            out[name] = value

        # TODO - add the possibility of non-huggingface metrics
        if external_metrics is not None:
            for i in range(len(external_metrics)):
                continue

        # Returns the results of the evaluation
        return out

    # Returns the compute_metrics functon
    return compute_metrics


###################################################################################################


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):

        # For int types
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        
        # For floats
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        
        # For arrays
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        
        # For anything else
        return json.JSONEncoder.default(self, obj)


###################################################################################################


def get_configs(config_path:str):
    """
    This loads the configurations from a file

    Args
        A string containing the path to the configuration file

    Output
        A dictionary that contains the configuration data
    """

    # Initialize the parser
    raw_config = configparser.RawConfigParser()   

    # Read the file
    raw_config.read(config_path)

    # Initialize config dictionary
    config = {}


    # Go through each section of the file
    for section in raw_config.sections():
        
        # We will load the items of the section into a dict
        config[section] = {}

        for key in raw_config[section].keys():

            # Decode the info using unicode
            value = raw_config[section][key].encode().decode('unicode_escape')

            # If the argument is a list, interpret it as a list
            if (value[0] == "[") and (value[-1] == "]"):
                for character in "\"\' ":
                    value = value.replace(character,"")
                value = value[1:-1].split(",")

            # If the value is supposed to be a number, pass it to its appropriate type
            elif value.isnumeric():
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)

            # If the value is boolean, interpret it as a boolean
            elif value.lower() in ["true", "ture"]:
                value = True

            elif value.lower() in ["false", "fasle"]:
                value = False
            
            # Save the fixed value
            config[section][key] = value

    return config


###################################################################################################


class CustomTrainer(Trainer):
    """
    This class adds the possibility to use a weighted loss function
    """


    # Initializer
    def __init__(self, class_weights, **kwargs):

        # Run the superclass initializer
        super(CustomTrainer, self).__init__(**kwargs)

        # Save the weights to be used
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)


    # Loss function
    def compute_loss(self, model, inputs, return_outputs=False):

        # Fetch the labels
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)

        # Extract the logits
        logits = outputs.get("logits")

        # Obtain the device for the logits
        device = logits.get_device()

        # Compute weighted loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(device))

        # Reshape the output
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


###################################################################################################


def handle_exception(exc_type, exc_value, exc_traceback):
    """ Fetches exceptions to log them """

    logger = logging.getLogger("uncaught_exception")

    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("The following exception was raised:",
                 exc_info=(exc_type, exc_value, exc_traceback))


###################################################################################################


# From https://github.com/huggingface/transformers/issues/3050#issuecomment-682167272
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix. It needs to be
    invoked after the modules have been loaded so that their loggers have been initialized.

    Args [optional]

        - level     desired level. e.g. logging.INFO
                    Default is logging.ERROR

        - prefices  list of one or more str prefices to match (e.g. ["transformers", "torch"])
                    Default is `[""]` to match all active loggers.
                    The match is a case-sensitive `module_name.startswith(prefix)`
    """

    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)