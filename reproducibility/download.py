import re
import os
import gzip
import shutil
import random
import requests
from tqdm.auto import tqdm
import pandas as pd
from rdkit import Chem


def download_file(url, output):
    """
    Downloads a file and log the progress

    (Strongly inspired from https://www.alpharithms.com/progress-bars-for-python-downloads-580122/)
    """

    # Make an HTTP request within a context manager
    with requests.get(url, stream=True) as r:
        if output.endswith("/"):
            if "Content-Disposition" in r.headers.keys():
                output += re.findall("filename=\"(.+)\"", r.headers["Content-Disposition"])[0]
            else:
                output += url.split("/")[-1]

        name = output
        if output.startswith('../data/assets/'):
            name = output[len('../data/assets/'):]

        if os.path.exists(output):
            print(f"Skipping {name} as it already exists")
            return

        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))

        # implement progress bar via tqdm
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc=name) as raw:
            # save the output to a file
            with open(output, 'wb') as output:
                shutil.copyfileobj(raw, output)


def unzip_file(input_file, output):
    # Decompress tar file
    with gzip.open(input_file) as f_in:
        with tqdm.wrapattr(open(output, 'wb'), "write", desc=f"Extracting {input_file}") as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_building_blocks():
    """
    Download .sdf file containing the molecule building blocks to train the model
    """

    url = 'https://drive.switch.ch/index.php/s/zLDApVjC7bU5qx2/download'

    path = '../data/assets/building-blocks/'

    download_file(url, path)


def download_ChEMBL(sample_size):
    """
    Download file containing a sample from ChEMBL database.

    Original source: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

    Args:
        sample_size: number of molecules to sample from full dataset
    """

    url = 'https://drive.switch.ch/index.php/s/jXuJyFIbADdSJkR/download'

    # Download and extract ChEMBL
    path = '../data/assets/molecules/'

    download_file(url, path)
    unzip_file(path + 'chembl_31.sdf.gz', path + 'chembl_31.sdf')

    # Read file and write a subset of SMILES
    with Chem.SDMolSupplier(path + 'chembl_31.sdf') as suppl:
        print("Extracting Mol samples from the file...")
        sample = [suppl[i] for i in tqdm(random.sample(range(len(suppl)), sample_size))]

    with open(path + 'chembl_smiles.txt', 'w') as f:
        for mol in sample:
            f.writelines(Chem.MolToSmiles(mol) + '\n')

    # delete decompressed file
    os.remove(path + 'chembl_31.sdf.gz')
    os.remove(path + 'chembl_31.sdf')


def download_ZINC(sample_size):
    """
    Download file containing a sample of drug-like molecules from ZINC15 database.

    Website: https://zinc.docking.org/tranches/home/
    """
    import random

    with open('../reproducibility/ZINC-downloader-2D-txt.uri', 'r') as f:
        links = f.read()
        links = links.split('\n')

    sublinks = random.sample(links, 20)

    master_df = pd.DataFrame()

    for link in sublinks:
        file_name = link[-8:]
        download_file(link, file_name)
        df = pd.read_csv(file_name, sep='\t')
        os.remove(file_name)
        master_df = pd.concat((master_df, df), axis=0)

    values = master_df['smiles'].sample(sample_size)

    with open('../data/assets/molecules/' + 'ZINC_SMILES.txt', 'w') as f:
        for value in values:
            f.write(value + '\n')


if __name__ == '__main__':
    samples = 10000

    print("\n[Building Blocks]")
    # download building blocks
    download_building_blocks()

    print("\n[ChEMBL Dataset]")
    # download a sample of 10000 molecules from ChEMBL and save them as SMILES
    download_ChEMBL(samples)  # slow

    print("\n[ZINC Dataset]")
    # download a sample of x molecules from ZINC database
    download_ZINC(samples)  # slow
