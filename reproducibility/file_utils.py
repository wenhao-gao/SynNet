import gzip
import shutil
from pathlib import Path

import tarfile
from tqdm.auto import tqdm


def file_name(uri):
    return uri.split("/")[-1]


def safe_remove(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False, onerror=None)
        else:
            path.unlink()


def should_skip(name: str, action: str, output_path: Path, force: bool) -> bool:
    """
    Define whether a file should be skipped or not

    The logic is simple : if it already exists, it is skipped.
    But, if the parameter 'force' is set to True, the file will never be skipped.

    Therefore, if the file exists but force is set, the file will be deleted

    :param name: name of the procedure
    :param action: action of the procedure
    :param output_path: path to check
    :param force: force the procedure
    :return: True if the procedure should be skipped
    """
    if output_path.exists():
        if force:
            print(f"{name} already exists. But 'force' is set, it will be re-{action}d")
            safe_remove(output_path)
        else:
            print(f"{name} is already present, no need to {action} it")
            return True
    return False


def extract_file(input_path: Path, output_path: Path, force: bool, extract_func):
    if should_skip(output_path.name, "extract", output_path, force):
        return

    size = input_path.stat().st_size
    with input_path.open("rb") as raw:
        # Wrap progress bar around file obj
        with tqdm.wrapattr(raw, 'read', total=size, desc=f"Extracting {input_path.name}") as f:
            extract_func(f)


def decompress_file(input_path: Path, output_path: Path, force: bool):
    def extract(f):
        # Decompress gz file
        with gzip.open(f) as f_in:
            with output_path.open('wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    extract_file(input_path, output_path, force, extract)


def extract_tar(input_path: Path, extract_root: Path, force: bool) -> Path:
    output_path = extract_root / (input_path.name.split('.')[0] + "_extracted")

    def extract(f):
        # Extract tar file
        with tarfile.open(fileobj=f) as tar:
            tar.extractall(output_path)

    extract_file(input_path, output_path, force, extract)
    return output_path
