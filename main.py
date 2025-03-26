import getopt
import os
import sys
import logging
import traceback
from translocdet.compute import run_translocation_detection
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(argv):
    config_filename = None
    logger = logging.getLogger()
    handler = logging.FileHandler(filename=os.path.join(os.path.dirname(__file__), "runtime.log"), mode='w', encoding='utf-8')
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s ; %(name)s ; %(levelname)s ; %(message)s",
                                            datefmt='%d/%m/%Y %H.%M'))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        opts, args = getopt.getopt(argv, "h:c:v:", ["Config=", "Verbose="])
    except getopt.GetoptError:
        print('usage: main.py -c <configuration_filepath> (--Verbose <mode>)')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -c <configuration_filepath> (--Verbose <mode>)')
            sys.exit()
        elif opt in ("-c", "--Config"):
            config_filename = arg
        elif opt in ("-v", "--Verbose"):
            if arg.lower() == 'debug':
                logger.setLevel(logging.DEBUG)
            elif arg.lower() == 'info':
                logger.setLevel(logging.INFO)
            elif arg.lower() == 'warning':
                logger.setLevel(logging.WARNING)
            elif arg.lower() == 'error':
                logger.setLevel(logging.ERROR)

    if not config_filename or not os.path.exists(config_filename):
        print('usage: main.py -c <config_filepath> (--Verbose <mode>)')
        sys.exit()

    try:
        run_translocation_detection(config_filename=config_filename)
    except Exception as e:
        logging.error(f'{e}')


if __name__ == "__main__":
    main(sys.argv[1:])
