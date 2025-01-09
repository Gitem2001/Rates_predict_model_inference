import yaml
import logging


logname = "debug_log.txt"

config_log = yaml.safe_load(open('./config/logger.yaml'))

logging.basicConfig(filename=config_log['basic_logger']['filename'],
                    filemode=config_log['basic_logger']['filemode'],
                    format=config_log['basic_logger']['format'],
                    datefmt=config_log['basic_logger']['datefmt'],
                    level=logging.DEBUG)

logger = logging.getLogger("basic_logger")



