# -*- coding: utf-8 -*-
import click
import os
import pandas as pd
import sys

from src.config import logger 

import logging
from kaggle.api.kaggle_api_extended import KaggleApi

# logging.basicConfig(level=logging.INFO)

def main():
    data_src = 'trolukovich/nutritional-values-for-common-foods-and-products'
    output_path = 'data/raw'
    logger.info(f"Downloading '{data_src}' from Kaggle.")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(data_src,force=True,unzip=True,path=output_path) 
    logger.info(f"Data saved at '{output_path}' directory.")

main()
# zf = ZipFile('nutritional-values-for-common-foods-and-products.zip')
# #extracted data is saved in the same directory as notebook
# zf.extractall() 
# zf.printdir()
# zf.close()