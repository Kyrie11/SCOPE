import logging

def get_logger(name='scope'):
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    return logging.getLogger(name)
