from tqdm import tqdm

from syn_net.utils.data_utils import Reaction


class BuildingBlockFilter:
    """Filter building blocks."""

    building_blocks: list[str]
    building_blocks_filtered: list[str]
    rxn_templates: list[str]
    rxns: list[Reaction]
    rxns_initialised: bool

    def __init__(
        self,
        *,
        building_blocks: list[str],
        rxn_templates: list[str],
        processes: int = 1,
        verbose: bool = False
    ) -> None:
        self.building_blocks = building_blocks
        self.rxn_templates = rxn_templates

        # Init reactions
        self.rxns = [Reaction(template=template.strip()) for template in self.rxn_templates]
        # Init other stuff
        self.processes = processes
        self.verbose = verbose
        self.rxns_initialised = False

    def _match_mp(self):
        from functools import partial

        from pathos import multiprocessing as mp

        def __match(bblocks: list[str], _rxn: Reaction):
            return _rxn.set_available_reactants(bblocks)

        func = partial(__match, self.building_blocks)
        with mp.Pool(processes=self.processes) as pool:
            self.rxns = pool.map(func, self.rxns)
        return self

    def _init_rxns_with_reactants(self):
        """Initializes a `Reaction` with a list of possible reactants.

        Info: This can take a while for lots of possible reactants."""
        self.rxns = tqdm(self.rxns) if self.verbose else self.rxns
        if self.processes == 1:
            [rxn.set_available_reactants(self.building_blocks) for rxn in self.rxns]
        else:
            self._match_mp()

        self.rxns_initialised = True
        return self

    def filter(self):
        """Filters out building blocks which do not match a reaction template."""
        if not self.rxns_initialised:
            self = self._init_rxns_with_reactants()
        matched_bblocks = {x for rxn in self.rxns for x in rxn.get_available_reactants}
        self.building_blocks_filtered = list(matched_bblocks)
        return self


from pathlib import Path


class BuildingBlockFileHandler:
    def _load_csv(self, file: str) -> list[str]:
        """Load building blocks as smiles from `*.csv` or `*.csv.gz`."""
        import pandas as pd

        return pd.read_csv(file)["SMILES"].to_list()

    def load(self, file: str) -> list[str]:
        """Load building blocks from file."""
        file = Path(file)
        if ".csv" in file.suffixes:
            return self._load_csv(file)
        else:
            raise NotImplementedError

    def _save_csv(self, file: Path, building_blocks: list[str]):
        """Save building blocks to `*.csv`"""
        import pandas as pd

        # remove possible 1 or more extensions, i.e.
        # <stem>.csv OR <stem>.csv.gz --> <stem>
        file_no_ext = file.parent / file.stem.split(".")[0]
        file = (file_no_ext).with_suffix(".csv.gz")
        # Save
        df = pd.DataFrame({"SMILES": building_blocks})
        df.to_csv(file, compression="gzip")
        return None

    def save(self, file: str, building_blocks: list[str]):
        """Save building blocks to file."""
        file = Path(file)
        if ".csv" in file.suffixes:
            self._save_csv(file, building_blocks)
        else:
            raise NotImplementedError
