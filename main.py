import yaml
import logging
import os
from datetime import datetime
import pandas as pd
from src.train import train_rnn_model

from src.logging_project import logger

from src.data_load import get_rates_between_dates, saving_data

def main():
    logger.info("Run app")
    # Configuration models
    config_path = "config/model.yaml"

    try:
        config = yaml.safe_load(open(config_path))
        logger.info(f"Successfully load the config file by path : {config_path}")
    except Exception as e:
        logger.critical(f"ERROR: {e}")
        exit(1)

    # Data update START
    data_files = os.listdir('data')
    if data_files != 0:
        logger.debug(data_files)
        start_date = pd.read_csv(f'data/{data_files[0]}')['date'].max()
        logger.info(f'Old data exist, starting data load process from {start_date} to {datetime.now().date()}')
        start_date = datetime.fromisoformat(start_date)
        data = get_rates_between_dates(start_date,
                                       datetime.now(),
                                       config['data']['currencies'])
        saving_data(data, f'data_{datetime.now().date()}')
    else:
        logger.info(
            f'old data doesnt exist. Starting data load process from {config['data']['start_date']} to {config['data']['end_date']}')
        data = get_rates_between_dates(datetime.fromisoformat(config['data']['start_date']),
                                       datetime.fromisoformat(config['data']['end_date']),
                                       config['data']['currencies'])
        saving_data(data, f'data_{datetime.now().date()}')
    # Data update END

    models = os.listdir('models')
    if len(models) != 0:
        # если модели есть
        logger.debug(models)
    else:
        logger.info('Models dont exist, starting data load process')
        train_rnn_model(data=data,
                        input_size=config['model']['input_size'],
                        hidden_layer_size=config['model']['hidden_layer_size'],
                        output_size=config['model']['output_size'],
                        num_layers=config['model']['num_layers'],
                        lr=config['train']['lr'],
                        num_epochs=config['train']['num_epoch'],
                        seq_len=config['data_prep']['seq_len'],
                        name_model=config['model']['name'])


if __name__ == "__main__":
    main()