import logging
import time

from .Processing.AI.run import run_ai
from .Processing.Classical.run import run_classical
from .Utils.configuration import ResourcesConfiguration


def run_translocation_detection(config_filename: str, logging_filename: str = None) -> None:
    """ """
    # Exceptions thrown by the parsing of the configuration file are on purpose outside the try/catch by design.
    ResourcesConfiguration.getInstance().set_environment(config_path=config_filename)
    if logging_filename:
        logger = logging.getLogger()
        handler = logging.FileHandler(filename=logging_filename, mode="a", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s ; %(name)s ; %(levelname)s ; %(message)s", datefmt="%d/%m/%Y %H.%M")
        )
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    logging.info("Starting the translocation detection process.")
    start = time.time()
    try:
        if ResourcesConfiguration.getInstance().default_method == "classical":
            run_classical()
        elif ResourcesConfiguration.getInstance().default_method == "ai":
            run_ai()
    except Exception as e:
        if ResourcesConfiguration.getInstance().default_error_handling == "log":
            logging.error("""[Backend error] Translocation detection failed with:\n{}""".format(e))
        elif ResourcesConfiguration.getInstance().default_error_handling == "break":
            raise ValueError(f"{e}")
    logging.info("Total elapsed time for executing the detection: {} seconds.".format(time.time() - start))
