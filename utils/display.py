import pandas as pd
import logging


def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    return


def configure_pandas():
    desired_width = 1280
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 20)
    return
