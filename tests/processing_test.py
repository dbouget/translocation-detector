import os
import shutil
import configparser
import logging
import sys
import subprocess
import traceback
import zipfile

try:
    import requests
    import gdown
    if int(gdown.__version__.split('.')[0]) < 4 or int(gdown.__version__.split('.')[1]) < 4:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown==4.4.0'])
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'requests==2.28.2'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'gdown==4.4.0'])
    import gdown
    import requests


def processing_test():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Running processing unit test.\n")
    logging.info("Downloading unit test resources.\n")
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_tests_results_dir')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    try:
        test_resources_url = None  # Has to be filled, e.g., 'https://github.com/.zip'

        archive_dl_dest = os.path.join(test_dir, 'test_resources.zip')
        headers = {}
        response = requests.get(test_resources_url, headers=headers, stream=True)
        response.raise_for_status()
        if response.status_code == requests.codes.ok:
            with open(archive_dl_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1048576):
                    f.write(chunk)
        with zipfile.ZipFile(archive_dl_dest, 'r') as zip_ref:
            zip_ref.extractall(test_dir)
    except Exception as e:
        logging.error("Error during resources download with: \n {}.\n".format(traceback.format_exc()))
        shutil.rmtree(test_dir)
        raise ValueError("Error during resources download.\n")

    logging.info("Preparing configuration file.\n")
    try:
        test_config = configparser.ConfigParser()
        test_config.add_section('Default')
        test_config.set('Default', 'method', "classical")
        test_config.set('Default', 'error_handling', "log")
        test_config.add_section('System')
        test_config.set('System', 'gpu_id', "-1")
        test_config.set('System', 'input_path', os.path.join(test_dir, 'inputs'))
        test_config.set('System', 'output_folder', test_dir)
        test_config.add_section('Classical')
        test_config.set('Classical', 'blobdetector_min_circularity', '0.7')
        test_config.set('Classical', 'blobdetector_min_convexity', '0.3')
        test_config.set('Classical', 'marker_intensity_threshold', '180')
        test_config_filename = os.path.join(test_dir, 'test_processing_config.ini')
        with open(test_config_filename, 'w') as outfile:
            test_config.write(outfile)

        logging.info("Processing CLI unit test started.\n")
        try:
            import platform
            if platform.system() == 'Windows':
                subprocess.check_call(['translocdet',
                                       '{config}'.format(config=test_config_filename),
                                       '--verbose', 'debug'], shell=True)
            elif platform.system() == 'Darwin' and platform.processor() == 'arm':
                subprocess.check_call(['python3', '-m', 'translocdet',
                                       '{config}'.format(config=test_config_filename),
                                       '--verbose', 'debug'])
            else:
                subprocess.check_call(['translocdet',
                                       '{config}'.format(config=test_config_filename),
                                       '--verbose', 'debug'])
        except Exception as e:
            logging.error(f"Error during processing CLI unit test with: {e}\n {traceback.format_exc()}.\n")
            shutil.rmtree(test_dir)
            raise ValueError("Error during processing CLI unit test.\n")

        logging.info("Collecting and comparing results.\n")
        results_filename = os.path.join(test_dir, 'file.csv')
        if not os.path.exists(results_filename):
            logging.error("Processing CLI unit test failed, <reason>.\n")
            shutil.rmtree(test_dir)
            raise ValueError("Processing CLI unit test failed, <reason>.\n")

        logging.info("Inference CLI unit test succeeded.\n")

        logging.info("Running processing unit test with Python package API.\n")
        from translocdet.compute import run_translocation_detection
        run_translocation_detection(config_filename=test_config_filename)

        logging.info("Collecting and comparing results.\n")
        results_filename = os.path.join(test_dir, 'file.csv')
        if not os.path.exists(results_filename):
            logging.error("Processing unit test with Python package API, <reason>.\n")
            raise ValueError("Processing unit test with Python package API, <reason>.\n")
        os.remove(results_filename)
    except Exception as e:
        logging.error(f"Running processing unit test with Python package API: {e}\n {traceback.format_exc()}.\n")
        shutil.rmtree(test_dir)
        raise ValueError("Running processing unit test with Python package API.\n")

    logging.info("Inference unit test succeeded.\n")

    shutil.rmtree(test_dir)


if __name__ == "__main__":
    processing_test()
