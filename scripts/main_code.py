"""
+------------------------------------------------------------------------------------------------+
|                                                                                                |
|                                         Code by RIMUSA                                         |
|                                                                                                |
|   This is the main code for the model trainers                                                 |
|   It consists of the arguments loader (main) and the model trainer function (train_model)      |
|                                                                                                |
+------------------------------------------------------------------------------------------------+
"""

#  Package imports
import importlib
import warnings
import datasets
import logging
import sys
import os


# Local imports
try:
    from .model import Generate_Model
    from .utils import get_configs, handle_exception, set_global_logging_level, text_to_list
except ImportError:
    from model import Generate_Model
    from utils import get_configs, handle_exception, set_global_logging_level, text_to_list


###################################################################################################


def train_model(model_args, trainer_args, data_args,
                train:bool=False, test:bool=False, predict:bool=False,
                save_model:str=None, save_results:str=None,
                load_model:str=None, load_file:str=None,
                embed:str=False, embed_column:str=None,
                device:str=None, metrics:list=None):
    """
    This is the main function to get models to run. In theory, all of the model, training, and data
    arguments should come packaged, while the other arguments should be more general options.
    
    Note that this function will do nothing unless you explicitly ask it to do something.

    
    Args [compulsory]

        - model_args    a dict-like object containing the arguments to initialize the model. Look
                        into the model.py file for more info on what this has to contain.

        - trainer_args  a dict-like object containing the arguments for the training loop. Look
                        into the model.py file for more info on what this has to contain. Note that
                        these contents might need to be changed depending on whether you're
                        training, testing, or doing inference.

        - data_args     a dict-like object containing the arguments for the dataset. Look into the
                        dataloaders.py file for more info on what this has to contain.


    Args [optional]

        -----------------------------------------TRAINING-----------------------------------------

        - train         a bool that determines whether the loaded model should be trained. If no
                        "training" split was given, it will use the first split.
                        Default: False

        - save_model    a string that contains the path where to save the model to. If the value
                        is None, the model won't be saved.
                        This only works if the train flag is up.
                        Default: None

        ------------------------------------------TESTING------------------------------------------

        - test          a bool that determines whether the loaded model should be evaluated. If no
                        "test" split was given, it will use the last split.
                        Default: False
                        
        - save_results  a string that contains the path where to save the results to. If the value
                        is None, the results won't be saved.
                        This only works if the test flag is up.
                        Default: None

        - metrics       a list with the metrics that should be used for testing.
                        The format for each entry should be:
                        name:str when using a metric with no averaging
                        ( name:str , average_method:str ) when using averaging
                        Default: None

        ----------------------------------------PREDICTION----------------------------------------

        - embed         a bool that determines whether the loaded model should generate sentence
                        embeddings for the dataset. Note that this is completely independent from
                        the predict argument.
                        Default: False

        - embed_column  a string that indicates the column from which the embeddings will be
                        generated from.
                        Default: None

        - predict       a bool that determines whether the loaded model should make predictions. If
                        no "predict" split was given, it will use the last split.
                        Default: False

        -------------------------------------------OTHER-------------------------------------------

        - load_model    a bool given to non-huggingface models to tell them to load a pretrained
                        model. This function is done automatically by the model_name argument in
                        model_args for huggingface models.
                        Default: False

        - load_file     a string given to non-huggingface models to tell them where to load a
                        pretrained model from. This function is done automatically by the
                        model_name argument in model_args for huggingface models.
                        Default: None

        - device        a string that should determine the device(s) that cuda will be using.
                        NOTE: As of 2023-09-23 this hasn't been implemented yet.
                        Default: None

    """

    # Initialize the model
    model = Generate_Model(model_args, trainer_args, data_args,
                           load_path=load_model, load_file=load_file)


    # If the training flag is up, train the model
    if train:

        # Obtain the training set
        if "training" in data_args["datasets"]:
            train_set = "training"
        else:
            train_set = data_args["datasets"][0]

        # Obtain the validation set
        if "validation" in data_args["datasets"]:
            eval_set = "validation"
        elif len(data_args["datasets"]) == 1:
            logging.warn("Be sure not to use the same dataset for training and validation!")
            eval_set  = data_args["datasets"][0]
        else:
            eval_set  = data_args["datasets"][1]

        # Start training
        model.train(train=train_set, eval=eval_set)

        # If the save flag is up, save the model
        if save_model is not None:
            model.save(save_path=save_model)


    # If the evaluation flag is up, evaluate the model
    if test:

        # Obtain the evaluation set
        if "test" in data_args["datasets"]:
            test_set = "test"
        else:
            test_set = data_args["datasets"][-1]

        # Load the evaluation metrics
        if metrics is not None:
            test_metrics = text_to_list(metrics[1:-1])
        else:
            test_metrics = ["accuracy", "f1", ("f1","macro"), ("f1", "weighted")]
       
        # Start evaluation
        model.evaluate(eval=test_set, metrics=test_metrics)

        # If the save results is up, save the evaluation
        if save_results is not None:
            model.save_evaluation(save_results)


    # If the prediction flag is up, generate predictions from data
    if predict:

        # Obtain the data for predictions
        if "predict" in data_args["datasets"]:
            predict_set = "predict"
        else:
            predict_set = data_args["datasets"][-1]

        # Start predicting
        model.predict(predict=predict_set)

    # If the embeddings flag is up, generate embeddings from the data
    # Note that it generates embeddings for all splits of the fataset
    if embed:
        model.sentence_embeddings(column=embed_column)

    return


###################################################################################################


def main():
    """
    Main function of the library
    """

    # Read the non-importable arguments
    config_file = sys.argv[1]
    
    # Load configs
    config = get_configs(config_file)

    # Load the different kinds of arguments
    args = config["data"]
    trainer_args = config["training"]
    data_args = config["dataset"]
    model_args = config["model"]

    # Activate or deactivate logging
    logging_path = ""
    if "logging_path" in config["logging"].keys():
        logging_path = config["logging"]["logging_path"]
        if "filemode" in config["logging"].keys():
            filemode = config["logging"]["filemode"]
        else:
            filemode = "a"
        os.makedirs(os.path.dirname(logging_path), exist_ok=True)
        logging.basicConfig(level=logging.INFO, filename=logging_path,
                            filemode=filemode, force=True)
        sys.excepthook = handle_exception

    # Only use this if you know your code does what you expect it to do!
    # It disables non-error logging from HugginFace, torch, and cuda
    # It also disables UserWarnings from torch.nn
    if (
        ("huggingface_logging" in config["logging"].keys())
        and not config["logging"]["huggingface_logging"]
       ):
        logging.warning("Logging and warning from Huggingface, torch, and cuda are turned off.")
        set_global_logging_level(logging.ERROR, ["transformers", "datasets", "torch", "cuda"])
        warnings.filterwarnings(action='ignore', category=UserWarning, module='torch.nn')
        datasets.disable_progress_bar() 

    # This part of the code runs several experiments if the argument is given
    if ("meta" in config.keys()) and ("n_experiments" in config["meta"].keys()):
        
        n_experiments = config["meta"]["n_experiments"]

        temp_args = {}

        # Generate saving and loading arguments
        for key in ["save_model", "save_results", "load_model", "load_file"]:
            if key in args.keys():
                temp_args[key] = args[key]
            else:
                temp_args[key] = None

        # Run the experiments
        for i in range(n_experiments):
            postfix = "_" + str(i)
            for key in temp_args.keys():
                if temp_args[key] is not None:
                    args[key] = temp_args[key] + postfix
                else:
                    args[key] = None

            logging.info("")
            logging.info("Starting experiment number {}!".format(i))
            train_model(model_args, trainer_args, data_args, **args)


    # This part of the code runs just a single experiment
    else:
        logging.info("")
        logging.info("Starting experiment!")
        train_model(model_args, trainer_args, data_args, **args)


    logging.info("All experiments run successfully!")
    logging.info("")


    return


###################################################################################################