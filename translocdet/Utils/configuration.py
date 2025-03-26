import configparser
import logging
from typing import List


class ResourcesConfiguration:
    """
    Singleton class to have access from anywhere in the code at the various paths and configuration parameters.
    """

    __instance = None
    __config_filename = None
    __config = None
    __accepted_image_formats = ["nd2", "png", "jpeg"]
    __default_method = None
    __default_error_handling = None
    __system_input_path = None
    __system_output_folder = None
    __classical_blobdetector_min_circularity = None
    __classical_blobdetector_min_convexity = None
    __classical_marker_intensity_threshold = None

    @staticmethod
    def getInstance():
        """Static access method."""
        if ResourcesConfiguration.__instance is None:
            ResourcesConfiguration()
        return ResourcesConfiguration.__instance

    def __init__(self):
        """Virtually private constructor."""
        if ResourcesConfiguration.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ResourcesConfiguration.__instance = self
            self.__setup()

    def __setup(self):
        """
        Definition of all attributes accessible through this singleton.
        """
        self.__config_filename = None
        self.__config = None

        self.__default_method = None
        self.__default_error_handling = "break"

        self.system_gpu_id = "-1"
        self.__system_input_path = None
        self.__system_output_folder = None

        self.__classical_blobdetector_min_circularity = 0.7
        self.__classical_blobdetector_min_convexity = 0.3
        self.__classical_marker_intensity_threshold = 180

        self.ai_models_folder = None

    @property
    def config_filename(self) -> str:
        return self.__config_filename

    @config_filename.setter
    def config_filename(self, fn: str) -> None:
        self.__config_filename = fn

    @property
    def config(self) -> configparser.ConfigParser:
        return self.__config

    @config.setter
    def config(self, conf: configparser.ConfigParser) -> None:
        self.__config = conf

    @property
    def accepted_image_formats(self) -> List[str]:
        return self.__accepted_image_formats

    @property
    def default_method(self) -> str:
        return self.__default_method

    @default_method.setter
    def default_method(self, m: str) -> None:
        self.__default_method = m

    @property
    def default_error_handling(self) -> str:
        return self.__default_error_handling

    @default_error_handling.setter
    def default_error_handling(self, m: str) -> None:
        self.__default_error_handling = m

    @property
    def system_input_path(self) -> str:
        return self.__system_input_path

    @system_input_path.setter
    def system_input_path(self, fn: str) -> None:
        self.__system_input_path = fn

    @property
    def system_output_folder(self) -> str:
        return self.__system_output_folder

    @system_output_folder.setter
    def system_output_folder(self, fn: str) -> None:
        self.__system_output_folder = fn

    @property
    def classical_blobdetector_min_circularity(self) -> float:
        return self.__classical_blobdetector_min_circularity

    @classical_blobdetector_min_circularity.setter
    def classical_blobdetector_min_circularity(self, val: float) -> None:
        self.__classical_blobdetector_min_circularity = val

    @property
    def classical_blobdetector_min_convexity(self) -> float:
        return self.__classical_blobdetector_min_convexity

    @classical_blobdetector_min_convexity.setter
    def classical_blobdetector_min_convexity(self, val: float) -> None:
        self.__classical_blobdetector_min_convexity = val

    @property
    def classical_marker_intensity_threshold(self) -> int:
        return self.__classical_marker_intensity_threshold

    @classical_marker_intensity_threshold.setter
    def classical_marker_intensity_threshold(self, val: int) -> None:
        self.__classical_marker_intensity_threshold = val

    def set_environment(self, config_path: str = None) -> None:
        """
        Iterating over the provided user configuration file to populate the internal variables.

        Parameters
        ----------
        config_path: str
            Full path to the configuration file provided by the user.

        Returns
        --------
        """
        try:
            self.config = configparser.ConfigParser()
            self.config_filename = config_path
            self.config.read(self.config_filename)
            self.__parse_default_parameters()
            self.__parse_system_parameters()
            self.__parse_classical_parameters()
            self.__parse_ai_parameters()
        except Exception as e:
            logging.error(f"Parsing configuration failed, received exception: {e}.")

    def __parse_default_parameters(self):
        eligible_methods = ["classical", "ai"]
        eligible_handlings = ["log", "break"]

        if self.config.has_option("Default", "method"):
            if self.config["Default"]["method"].split("#")[0].strip() != "":
                self.default_method = self.config["Default"]["method"].split("#")[0].strip()
        if self.default_method not in eligible_methods:
            raise AttributeError(
                f"Requested method {self.default_method} not eligible. " f"Please choose within: {eligible_methods}"
            )

        if self.config.has_option("Default", "error_handling"):
            if self.config["Default"]["error_handling"].split("#")[0].strip() != "":
                self.default_error_handling = self.config["Default"]["error_handling"].split("#")[0].strip()
        if self.default_error_handling not in eligible_handlings:
            raise AttributeError(
                f"Requested error handling {self.default_error_handling} not eligible. "
                f"Please choose within: {eligible_handlings}"
            )

    def __parse_system_parameters(self):
        if self.config.has_option("System", "input_path"):
            if self.config["System"]["input_path"].split("#")[0].strip() != "":
                self.system_input_path = self.config["System"]["input_path"].split("#")[0].strip()

        if self.config.has_option("System", "output_folder"):
            if self.config["System"]["output_folder"].split("#")[0].strip() != "":
                self.system_output_folder = self.config["System"]["output_folder"].split("#")[0].strip()

    def __parse_classical_parameters(self):
        if self.config.has_option("Classical", "blobdetector_min_circularity"):
            if self.config["Classical"]["blobdetector_min_circularity"].split("#")[0].strip() != "":
                self.classical_blobdetector_min_circularity = float(
                    self.config["Classical"]["blobdetector_min_circularity"].split("#")[0].strip()
                )
        if self.classical_blobdetector_min_circularity < 0.0 or self.classical_blobdetector_min_circularity > 1.0:
            raise ValueError(
                f"['classical']['blobdetector_min_circularity'] must be between 0 and 1, "
                f"{self.classical_blobdetector_min_circularity} is incompatible."
                f"Please adjust the configuration file!"
            )

        if self.config.has_option("Classical", "blobdetector_min_convexity"):
            if self.config["Classical"]["blobdetector_min_convexity"].split("#")[0].strip() != "":
                self.classical_blobdetector_min_convexity = float(
                    self.config["Classical"]["blobdetector_min_convexity"].split("#")[0].strip()
                )
        if self.classical_blobdetector_min_convexity < 0.0 or self.classical_blobdetector_min_convexity > 1.0:
            raise ValueError(
                f"['classical']['blobdetector_min_convexity'] must be between 0 and 1, "
                f"{self.classical_blobdetector_min_convexity} is incompatible."
                f" Please adjust the configuration file!"
            )

        if self.config.has_option("Classical", "marker_intensity_threshold"):
            if self.config["Classical"]["marker_intensity_threshold"].split("#")[0].strip() != "":
                self.classical_marker_intensity_threshold = float(
                    self.config["Classical"]["marker_intensity_threshold"].split("#")[0].strip()
                )
        if self.classical_marker_intensity_threshold < 0 or self.classical_marker_intensity_threshold > 255:
            raise ValueError(
                f"['classical']['marker_intensity_threshold'] must be between 0 and 255,"
                f" {self.classical_marker_intensity_threshold} is incompatible."
                f" Please adjust the configuration file!"
            )

    def __parse_ai_parameters(self):
        pass
