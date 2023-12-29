import os
import subprocess
import click
import emv.utils

from pathlib import Path
from typing import List, Union, Optional

LOG = emv.utils.get_logger()

# List of valid image extensions
VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']


def list_images_in_folder(folder_path: str) -> List[str]:
    """List valid image files in the specified folder."""
    images = []
    for file in os.listdir(folder_path):
        if Path(file).suffix.lower() in VALID_EXTENSIONS:
            images.append(os.path.join(folder_path, file))
    return images


def compress_images(input_images: Union[str, List[str]],
                    output_folder: Optional[str] = None,
                    options: Optional[List[str]] = None,
                    binary_path: str = '/usr/local/bin/nvcompress') -> None:
    # Ensure the binary exists
    if not Path(binary_path).exists():
        raise FileNotFoundError(f"The binary {binary_path} does not exist.")

    # Ensure input_images is a list
    if isinstance(input_images, str):
        input_images = [input_images]

    for input_image in input_images:
        # Determine the output folder
        if output_folder:
            output_dir = Path(output_folder)
        else:
            output_dir = Path(input_image).parent

        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the output file path
        output_file = output_dir / Path(input_image).name

        cmd = [binary_path]
        if options:
            cmd.extend(options)
        cmd.append("-silent")
        cmd.extend([input_image, str(output_file)])

        try:
            _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
            LOG.info(f"{input_image} compressed")
        except subprocess.CalledProcessError as e:
            LOG.error(f"Error for {input_image}: {e.stderr}")


@click.command()
@click.option('-i', '--input-folder', default=None, type=str, help='Path to the input folder containing images')
@click.option('-f', '--output-folder', default=None, type=str, help='Path to the output folder (optional)')
@click.option('-o', '--options', multiple=True, type=str, help='Additional options for nvcompress')
@click.option('-b', '--binary-path', default='/usr/local/bin/nvcompress', type=str, help='Path to the nvcompress binary')
@click.argument('input_images', nargs=-1, required=False)
def main(input_folder, output_folder, options, binary_path, input_images):
    if input_folder:
        input_images = list_images_in_folder(input_folder)
        if not input_images:
            LOG.warning("No valid image files found in the input folder.")
            return
    elif not input_images:
        LOG.error("No input images specified.")
        return

    compress_images(input_images, output_folder, list(options), binary_path)


if __name__ == '__main__':
    main()