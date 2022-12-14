import gzip
import shutil
from pathlib import Path

import tarfile

import pandas as pd
from tqdm.auto import tqdm

from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt
from synnet.models.mlp import MLP

smile = str


def file_name(uri: str) -> str:
    """
    Extract the filename from a URI

    Args:
        uri: URI to extract name from. Expected to end with the filename

    Returns:
        The file name
    """
    return uri.split("/")[-1]


def safe_remove(path: Path):
    """Delete a file safely

    Make sure a file at given path is removed.
    It removes any directory recursively and only applies if the file exists.

    Args:
        path: Path to file to remove
    """
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

    Args:
        name: name of the procedure
        action: action of the procedure
        output_path: path to check
        force: force the procedure

    Returns:
        True if the procedure should be skipped
    """
    if output_path.exists():
        if force:
            print(f"{name} already exists. But 'force' is set, bypassing...")
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
    """
    Decompress a gzip file

    If the output file already exists, this will be skipped
    This can be bypassed by setting force to True

    Args:
        input_path: Path to the gzip file
        output_path: Path to the decompressed file
        force: If set to true, the file will be decompressed even if it has already been done
    """
    def extract(f):
        # Decompress gz file
        with gzip.open(f) as f_in:
            with output_path.open('wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    extract_file(input_path, output_path, force, extract)


def extract_tar(input_path: Path, extract_root: Path, force: bool) -> Path:

    """
    Extract a tarball file

    If the output file already exists, this will be skipped
    This can be bypassed by setting force to True

    Args:
        input_path: Path to the gzip file
        extract_root: Directory to output extracted files
        force: If set to true, the file will be decompressed even if it has already been done

    Returns:
        The parent path to all output files
    """
    output_path = extract_root / (input_path.name.split('.')[0] + "_extracted")

    def extract(f):
        # Extract tar file
        with tarfile.open(fileobj=f) as tar:
            tar.extractall(output_path)

    extract_file(input_path, output_path, force, extract)
    return output_path


def save_smiles(smiles: list[smile], output_path: Path):
    """
    Store a smile list as a csv.gz

    Args:
        smiles (list[str]): The smiles to store
        output_path (Path): File path
    """
    df = pd.DataFrame(smiles, columns=["smiles"])
    df.to_csv(output_path, index=False, compression="gzip")


def load_smiles(input_path: Path) -> list[smile]:
    """
    Load smiles from a file

    Args:
        input_path (Path): File path

    Returns:
        A list containing the smiles
    """
    return pd.read_csv(input_path)["smiles"]


def load_checkpoints(checkpoint_path: Path) -> list[MLP]:
    """
    Load the models' checkpoints given a path

    Args:
        checkpoint_path (Path): Directory where the checkpoints are stored

    Returns:
        A list containing the checkpoints
    """
    ckpt_files = [find_best_model_ckpt(str(checkpoint_path / model)) for model in "act rt1 rxn rt2".split()]
    return [load_mlp_from_ckpt(str(file)) for file in ckpt_files]
