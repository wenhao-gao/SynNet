import wget
import os

def download_building_blocks():
    '''Download file containing the molecule building blocks to train the model

    '''

    url = 'https://drive.switch.ch/index.php/s/zLDApVjC7bU5qx2/download'
    
    path = 'data/assets/building-blocks/'
    
    file = wget.download(url, out=path)
    


if __name__ == '__main__':

    download_building_blocks()