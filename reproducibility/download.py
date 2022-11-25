import wget
import os
import gzip
import shutil
import random
import pandas as pd
from rdkit import Chem

def download_building_blocks():
    '''Download .sdf file containing the molecule building blocks to train the model

    '''

    url = 'https://drive.switch.ch/index.php/s/zLDApVjC7bU5qx2/download'
    
    path = 'data/assets/building-blocks/'
    
    file = wget.download(url, out=path)



def download_ChEMBL(sample_size):
    '''Download file containing a sample from ChEMBL database. 
    
    Original source: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/
    
    Args:
        -n: number of molecules to sample from full dataset

    '''
    
    url = 'https://drive.switch.ch/index.php/s/jXuJyFIbADdSJkR/download'
    
    #Download and extract ChEMBL
    path = 'data/assets/molecules/'

    file = wget.download(url, out=path)
    
    # Decompress tar file
    with gzip.open(path + 'chembl_31.sdf.gz', 'rb') as f_in:
        with open(path + 'chembl_31.sdf', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    #Read file and write a subset of SMILES
    suppl = Chem.SDMolSupplier(path + 'chembl_31.sdf')
    
    mols = [x for x in suppl]

    sample = random.sample(mols, sample_size)
    

    with open(path + 'chembl_smiles.txt', 'w') as f:
        for mol in sample:
            f.writelines(Chem.MolToSmiles(mol) + '\n')

    #Delete decompressed file
    os.remove(path + 'chembl_31.sdf.gz')
    os.remove(path + 'chembl_31.sdf')


def download_ZINC(sample_size):
    '''Download file containing a sample of drug-like molecules from ZINC15 database.
    Website: https://zinc.docking.org/tranches/home/
    
    '''
    import random 
    
    with open('reproducibility/ZINC-downloader-2D-txt.uri', 'r') as f:
        links = f.read()
        links = links.split('\n')

    sublinks = random.sample(links, 20)

    master_df = pd.DataFrame()

    for link in sublinks:
        file = wget.download(link)
        df = pd.read_csv(link[-8:], sep ='\t')
        os.remove(link[-8:])
        master_df = pd.concat((master_df,df), axis=0)

    values = master_df['smiles'].sample(sample_size)

    with open('data/assets/molecules/' + 'ZINC_SMILES.txt', 'w') as f:
        for value in values:
            f.write(value + '\n')



if __name__ == '__main__':
    
    #download building blocks
    download_building_blocks()

    #download a sample of 10000 molecules from ChEMBL and save them as SMILES
    download_ChEMBL(10000) #slow
    
    #download a sample of x molecules from ZINC database
    download_ZINC(10000) #slow
