# coding=utf-8
#!/user/bin/env python3

import logging.config
import os

import coloredlogs
import yaml

_logger = logging.getLogger('')
_app_name = os.getenv('APP_NAME', None)
_path = 'resources/logger.yml'
_extra = {'app_name': _app_name}

if os.path.exists(_path):
    with open(_path, 'rt') as f:
        try:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
            coloredlogs.install(fmt=config['formatters']['standard']['format'],
                                level=config['handlers']['console']['level'])
        except Exception as e:
            print(e)
            print('Error in Logging Configuration. Using default configs')
            logging.basicConfig(level=logging.INFO)
            coloredlogs.install(level=logging.INFO)
else:
    logging.basicConfig(level=logging.DEBUG)
    coloredlogs.install(level=logging.DEBUG)
    print('Failed to load configuration file. Using default configs')

logger = logging.LoggerAdapter(_logger, _extra)
