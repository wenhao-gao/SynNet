import re
import os
import gzip
import shutil
import random
import requests
import tarfile
from tqdm.auto import tqdm
import pandas as pd
from rdkit import Chem


def file_name(uri):
    return uri.split("/")[-1]


def download_file(url, output, progress_bar=True):
    """
    Downloads a file and log the progress

    (Inspired from https://www.alpharithms.com/progress-bars-for-python-downloads-580122/)
    """

    # Make an HTTP request within a context manager
    with requests.get(url, stream=True) as r:
        # Retrieve file name and update output if needed
        if os.path.isdir(output):
            if "Content-Disposition" in r.headers.keys():
                name = re.findall("filename=\"(.+)\"", r.headers["Content-Disposition"])[0]
            else:
                name = file_name(url)

            output += name
        else:
            name = file_name(output)

        if os.path.exists(output):
            print(f"Skipping {name} as it already exists")
            return

        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))

        stream = r.raw
        if progress_bar:
            # implement progress bar via tqdm
            stream = tqdm.wrapattr(r.raw, "read", total=total_length, desc=f"Downloading {name}")

        with stream as raw:
            # save the output to a file
            with open(output, 'wb') as output:
                shutil.copyfileobj(raw, output)


def safe_remove(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=False, onerror=None)
        else:
            os.remove(path)


def cleanup(*leftover_files):
    for f in leftover_files:
        safe_remove(f)


def should_skip(name, output_path, force) -> bool:
    if os.path.exists(output_path):
        if force:
            print(f"The {name} already exists. Forcefully re-downloading it")
            safe_remove(output_path)
        else:
            print(f"Skipping as {output_path} is already present")
            return True
    return False


def decompress_file(input_file, output_file):
    size = os.stat(input_file).st_size
    with open(input_file, "rb") as raw:
        # Wrap progress bar around file obj
        with tqdm.wrapattr(raw, 'read', total=size, desc=f"Extracting {file_name(input_file)}") as f:
            # Decompress gz file
            with gzip.open(f) as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def extract_file(input_file, output_file):
    size = os.stat(input_file).st_size
    with open(input_file, "rb") as raw:
        # Wrap progress bar around file obj
        with tqdm.wrapattr(raw, 'read', total=size, desc=f"Extracting {file_name(input_file)}") as f:
            # Extract tar file
            with tarfile.open(fileobj=f) as tar:
                tar.extractall(output_file)


def convert_sdf_to_csv_gz(input_file, output_file, sample=None):
    print("Loading sdf...  ", end="")
    supplier = Chem.SDMolSupplier(input_file)
    size = len(supplier)
    r = range(size)
    if sample is not None:
        size = sample
        r = random.sample(r, sample)

    # create mol generator and setup progress bar using tqdm
    mols = (supplier[i] for i in tqdm(r, total=size, desc="Converting sdf to smile", unit="mol"))
    smiles = (Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) for mol in mols)

    print("done")

    write_smiles_to_csv(output_file, smiles)


def write_smiles_to_csv(output_file, smiles):
    # write csv.gz file
    with gzip.open(output_file, "wt") as f:
        f.writelines("smiles\n")  # set csv column name

        for smile in smiles:
            f.write(smile + '\n')


def download_building_blocks(project_root, force=False):
    """
    Download .sdf file containing the molecule building blocks to train the model
    """

    sdf_file = project_root + "data/assets/building-blocks/enamine-us.sdf"
    output_file = project_root + "data/assets/building-blocks/enamine-us-smiles.csv.gz"

    url = 'https://drive.switch.ch/index.php/s/zLDApVjC7bU5qx2/download'

    if should_skip("building_blocks", sdf_file, force):
        return

    # cleanup potential residual files that could break the process
    cleanup(sdf_file)

    # download building_block sdf
    download_file(url, sdf_file)
    # convert sdf to smile csv
    convert_sdf_to_csv_gz(sdf_file, output_file)

    # cleaning up residual files
    cleanup(sdf_file)


def download_chembl(project_root, sample_size=None, force=False):
    """
    Download file containing a sample from ChEMBL database.

    Original source: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

    Args:
        sample_size: number of molecules to sample from full dataset
    """

    path = project_root + "data/assets/molecules/"
    compressed_sdf = path + "chembl_31.sdf.gz"
    sdf_file = path + "chembl_31.sdf"
    output_file = path + "chembl_31-smiles.csv.gz"

    url = 'https://drive.switch.ch/index.php/s/jXuJyFIbADdSJkR/download'

    # Check that it does not exist yet
    if should_skip("ChEMBL", output_file, force):
        return

    # cleanup potential residual files that could break the process
    cleanup(compressed_sdf, sdf_file)

    # Download and extract ChEMBL sdf
    download_file(url, path)
    decompress_file(compressed_sdf, sdf_file)

    # Convert sdf to smile csv
    convert_sdf_to_csv_gz(sdf_file, output_file, sample=sample_size)

    # cleaning up residual files
    cleanup(compressed_sdf, sdf_file)


def download_zinc(project_root, sample_size=None, force=False):
    """
    Download file containing a sample of drug-like molecules from ZINC15 database.

    Website: https://zinc.docking.org/tranches/home/
    """

    urls_path = project_root + "reproducibility/ZINC-downloader-2D.csv"
    output_path = project_root + "data/assets/molecules/zinc_smiles.csv.gz"

    # Check that it does not exist yet
    if should_skip("ZINC", output_path, force):
        return

    links = pd.read_csv(urls_path, header=None)

    if sample_size is not None:
        # Shuffle the link list such that we take samples from randomly chosen links
        # Use a seed generated by python's random, that way it depends on this random seed
        #
        # This is not perfect as the correlation between molecules in the same file is higher.
        # But it avoids downloading the whole dataset
        links = links.sample(frac=1, random_state=random.randint(0, 2**32-1))

    # Retrieve elements from the column
    links = links[0]

    master_df = pd.DataFrame()
    temp_file = "temp.csv"

    pbar = tqdm(total=sample_size)  # Create a manual progress bar

    for link in links:
        # Update progress bar
        pbar.set_description(f"Downloading {file_name(link)}")
        pbar.update(len(master_df.axes[0]))

        # Retrieve smiles from the link
        download_file(link, temp_file, progress_bar=False)
        df = pd.read_csv(temp_file, sep='\t')
        os.remove(temp_file)

        master_df = pd.concat((master_df, df), axis=0)
        # Stop if we have enough molecules
        if len(master_df.axes[0]) > sample_size:
            pbar.update(sample_size)
            break

    # Get only the smiles, and exactly the number we want
    smiles = master_df['smiles'].sample(sample_size)
    # Write it into a file
    write_smiles_to_csv(output_path, smiles)


def download_checkpoints(project_root, force=False):
    tar_file = "checkpoints.tar.gz"
    checkpoint_path = project_root + "checkpoints"
    checkpoint_temp_path = "hb_fp_2_4096_256"

    # Check that it does not exist yet
    if should_skip("checkpoints", checkpoint_path, force):
        return

    # cleanup potential residual files that could break the process
    cleanup(tar_file, checkpoint_temp_path)

    # retrieve the checkpoint tar file
    download_file("https://figshare.com/ndownloader/files/31067692", tar_file)
    # extract it
    extract_file(tar_file, '.')

    # move checkpoints to their respective folders
    os.mkdir(checkpoint_path)

    for model in ["act", "rt1", "rxn", "rt2"]:
        os.mkdir(f"{checkpoint_path}/{model}")
        shutil.move(
            f"./{checkpoint_temp_path}/{model}.ckpt",
            f"{checkpoint_path}/{model}/ckpts.dummy-val_loss=0.00.ckpt"
        )

    # cleaning up residual files
    cleanup(tar_file, checkpoint_temp_path)
    print("done")
