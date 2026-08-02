import logging


def get_logger(name):

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        console = logging.StreamHandler()

        formatter = logging.Formatter(

            "[%(levelname)s] %(message)s"

        )

        console.setFormatter(formatter)

        logger.addHandler(console)

    return logger