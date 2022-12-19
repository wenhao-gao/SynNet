import logging
import tempfile
from pathlib import Path
from typing import Optional, Union

import rdkit.Chem as Chem
from PIL import ImageOps

logger = logging.getLogger(__file__)


class MolDrawer:
    """Draws molecules as images."""

    def __init__(self, filetype: str = "png", tmpdir: Optional[str] = None):

        if filetype not in ["png"]:
            raise NotImplementedError()  # TODO: Add svg, scaling is difficult with png
        if tmpdir is None:
            tmpdir = tempfile.mkdtemp()

        # Init outfolder
        if not Path(tmpdir).exists():
            Path(tmpdir).mkdir(parents=True)
        logger.debug(f"Temporary directory set to: {tmpdir}")

        self.filetype = filetype
        self.tmpdir = tmpdir
        self._lookup: dict[str, str] = {}  # smiles -> filename

    def _plot_png(self, mol: Chem.Mol, filename: str, crop: bool = True, **kwargs):
        # Plot to PIL image
        img = Chem.Draw.MolToImage(mol, kwargs.get("size", (300, 300)))

        # Crop
        if crop:
            # https://stackoverflow.com/questions/9983263/how-to-crop-an-image-using-pil
            gmi = ImageOps.invert(img)  # invert white[255,255,255]->black[0,0,0]
            (left, upper, right, lower) = gmi.getbbox()  # calcutes non-zero bbox hence inverted
            img = ImageOps.crop(img, (left, upper, img.width - right, img.height - lower))

        # Save
        img.save(filename)
        return img

    def plot(self, smiles: Union[list[str], str], size=(300, 300)):
        """Plot smiles as 2d molecules and save to file."""
        if isinstance(smiles, str):
            smiles = [smiles]

        for smi in smiles:
            _, file = tempfile.mkstemp(dir=self.tmpdir, suffix="." + self.filetype)
            logger.debug(f"Will create {file}..")

            mol = Chem.MolFromSmiles(smi, sanitize=True)
            if not mol:
                logger.debug(f"Invalid smiles: {smi}")
                self._lookup[smi] = None
                continue

            # Plot
            if self.filetype == "png":
                self._plot_png(mol, file, size=size)

            # Store reference smiles -> filename
            self._lookup[smi] = file

        return self


if __name__ == "__main__":
    pass
