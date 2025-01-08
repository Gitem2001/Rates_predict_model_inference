import yaml
from src.logging_project import logger

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


if __name__ == "__main__":
    main()