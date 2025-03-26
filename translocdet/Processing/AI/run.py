import os

from translocdet.Utils.configuration import ResourcesConfiguration


def run_ai() -> None:
    """
    Main entry point for a translocation detection pipeline that will be AI-based.
    :return:
    """
    try:
        user_input = ResourcesConfiguration.getInstance().system_input_path
        output_folder = ResourcesConfiguration.getInstance().system_output_folder
        os.makedirs(output_folder, exist_ok=True)

        if os.path.exists(user_input) and not os.path.isdir(user_input):
            processing(input_file=user_input, output_folder=output_folder)
        elif os.path.isdir(user_input):
            input_images = []
            for _, _, files in os.walk(user_input):
                for f in files:
                    if f.endswith(tuple(ResourcesConfiguration.getInstance().accepted_image_formats)):
                        input_images.append(os.path.join(user_input, f))
                break

            for img_fn in input_images:
                processing(input_file=img_fn, output_folder=output_folder)
    except Exception as e:
        raise ValueError(f"AI computation failed with {e}")


def processing(input_file: str, output_folder: str):
    pass
