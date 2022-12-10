import re
import gzip
import shutil
import random
from pathlib import Path

import requests
import tarfile
from tqdm.auto import tqdm
import pandas as pd
from rdkit import Chem

from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.models.common import find_best_model_ckpt, load_mlp_from_ckpt

intermediates_path = Path('.') / "intermediates"
download_path = intermediates_path / "downloads"

intermediates_path.mkdir(exist_ok=True)
download_path.mkdir(exist_ok=True)


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


def download_file(url: str, output: Path, force: bool, progress_bar=True) -> Path:
    # Make an HTTP request within a context manager
    with requests.get(url, stream=True) as r:
        # Retrieve file name and update output if needed
        if output.is_dir():
            if "Content-Disposition" in r.headers.keys():
                name = re.findall("filename=\"(.+)\"", r.headers["Content-Disposition"])[0]
            else:
                name = file_name(url)

            output /= name
        else:
            name = output.name

        if should_skip(name, "downloade", output, force):
            return output

        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))

        stream = r.raw
        if progress_bar:
            # implement progress bar via tqdm
            stream = tqdm.wrapattr(r.raw, "read", total=total_length, desc=f"Downloading {name}")

        with stream as raw:
            # save the output to a file
            with output.open('wb') as out:
                shutil.copyfileobj(raw, out)

        return output


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


def convert_sdf_to_smiles(input_path: Path, sample=None):
    print("Loading sdf...  ", end="")
    supplier = Chem.SDMolSupplier(str(input_path))
    size = len(supplier)
    r = range(size)
    if sample is not None:
        size = sample
        r = random.sample(r, sample)
    print("done")

    # create mol generator and setup progress bar using tqdm
    mols = (supplier[i] for i in tqdm(r, total=size, desc="Converting sdf to smile", unit="mol"))
    return [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) for mol in mols]


def save_smiles(smiles, output_path):
    with output_path.open("w") as f:
        for smile in smiles:
            f.write(smile + "\n")


def load_smiles(output_path):
    with output_path.open("r") as f:
        return [line.strip() for line in f]


def get_building_blocks(project_root: Path, force=False) -> list[str]:
    """
    Download .sdf file containing the molecule building blocks to train the model
    """

    output_path = project_root / "data" / "assets" / "building-blocks" / "enamine-us-smiles.csv.gz"

    if should_skip("building_blocks", "compute", output_path, force):
        smiles = BuildingBlockFileHandler().load(str(output_path))
        return smiles

    # download building_block sdf
    sdf_file = download_file("https://drive.switch.ch/index.php/s/zLDApVjC7bU5qx2/download", download_path, force)
    # convert sdf to smiles
    smiles = convert_sdf_to_smiles(sdf_file)
    # save it
    BuildingBlockFileHandler().save(str(output_path), smiles)

    return smiles


def get_chembl_dataset(project_root: Path, sample_size=None, force=False) -> list[str]:
    """
    Download file containing a sample from ChEMBL database.

    Original source: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

    Args:
        sample_size: number of molecules to sample from full dataset
    """

    sdf_file = intermediates_path / "chembl_31.sdf"
    output_path = project_root / "data" / "assets" / "molecules" / "chembl-smiles.txt"

    # Check that it does not exist yet
    if should_skip("ChEMBL", "compute", output_path, force):
        # Load the data
        smiles = load_smiles(output_path)
        if sample_size and len(smiles) != sample_size:
            print("The number of samples differs from the expected amount. The dataset will be recomputed")
            safe_remove(output_path)
        else:
            return smiles

    # Download and extract ChEMBL sdf
    compressed_sdf = download_file("https://drive.switch.ch/index.php/s/jXuJyFIbADdSJkR/download", download_path, force)
    decompress_file(compressed_sdf, sdf_file, force)

    # Convert sdf to smiles
    smiles = convert_sdf_to_smiles(sdf_file, sample=sample_size)
    # save it
    save_smiles(smiles, output_path)

    return smiles


def get_zinc_dataset(project_root: Path, sample_size=None, force=False) -> list[str]:
    """
    Download file containing a sample of drug-like molecules from ZINC15 database.

    Website: https://zinc.docking.org/tranches/home/
    """

    urls_path = project_root / "reproducibility" / "ZINC-downloader-2D.csv"
    output_path = project_root / "data" / "assets" / "molecules" / "zinc_smiles.txt"

    # Check whether this process should be skipped
    if should_skip("ZINC", "compute", output_path, force):
        # Load the data
        smiles = load_smiles(output_path)
        if sample_size and len(smiles) != sample_size:
            print("The number of samples differs from the expected amount. The dataset will be recomputed")
            safe_remove(output_path)
        else:
            return smiles

    # Download the dataset
    links = pd.read_csv(urls_path, header=None)

    if sample_size is not None:
        # Shuffle the link list such that we take samples from randomly chosen links
        # Use a seed generated by python's random, that way it depends on this random seed
        #
        # This is not perfect as the correlation between molecules in the same file is high.
        # But it avoids downloading the whole dataset
        links = links.sample(frac=1, random_state=random.randint(0, 2**32-1))

    # Retrieve elements from the column
    links = links[0]

    master_df = pd.DataFrame()

    pbar = tqdm(total=sample_size)  # Create a manual progress bar

    zinc_downloads = download_path / "zinc"
    zinc_downloads.mkdir(exist_ok=True)

    for link in links:
        name = file_name(link)
        # Update progress bar
        pbar.set_description(f"Downloading {name}")
        pbar.update(master_df.shape[0])

        path = zinc_downloads / name

        # Retrieve smiles from the link
        download_file(link, path, force, progress_bar=False)
        df = pd.read_csv(path, sep='\t')

        master_df = pd.concat((master_df, df), axis=0)
        # Stop if we have enough molecules
        if master_df.shape[0] > sample_size:
            pbar.update(sample_size)
            break

    # Get only the smiles, and exactly the number we want
    smiles = master_df['smiles'].sample(sample_size)
    # Write it into a file
    save_smiles(smiles, output_path)

    return smiles


def load_checkpoints(checkpoint_path: Path):
    ckpt_files = [find_best_model_ckpt(str(checkpoint_path / model)) for model in "act rt1 rxn rt2".split()]
    return [load_mlp_from_ckpt(str(file)) for file in ckpt_files]


def get_original_checkpoints(project_root: Path, force=False) -> list:
    tar_file = download_path / "original_checkpoints.tar.gz"
    checkpoint_path = project_root / "original_checkpoints"

    # Check that it does not exist yet
    if should_skip("original_checkpoints", "compute", checkpoint_path, force):
        return load_checkpoints(checkpoint_path)

    # retrieve the checkpoint tar file
    download_file("https://figshare.com/ndownloader/files/31067692", tar_file, force)
    # extract it
    extract_output = extract_tar(tar_file, intermediates_path, force)
    checkpoint_temp_path = extract_output / "hb_fp_2_4096_256"

    # move original_checkpoints to their respective folders
    checkpoint_path.mkdir()

    for model in ["act", "rt1", "rxn", "rt2"]:
        model_path = checkpoint_path / model
        model_path.mkdir()

        shutil.move(
            checkpoint_temp_path / f"{model}.ckpt",
            model_path / "ckpts.dummy-val_loss=0.00.ckpt"
        )

    return load_checkpoints(checkpoint_path)
