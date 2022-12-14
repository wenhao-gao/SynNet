"""
This script is a rewriting of
 - script/01-filter-building-blocks.py
 - script/02-compute-embeddings.py

such that it can be used in a notebook.

The functions also verifies that the data has not been processed already.
"""
from pathlib import Path

from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    BuildingBlockFilter,
    ReactionTemplateFileHandler,
)
from file_utils import should_skip, safe_remove, smile
from synnet.utils.data_utils import ReactionSet
from synnet.encoding.fingerprints import mol_fp
from synnet.MolEmbedder import MolEmbedder

from functools import partial


def filter_bblocks(project_root: Path, bblocks: list[smile], force=False) -> (list[smile], ReactionSet):
    """
    This function is equivalent to script/01-filter-building-blocks.py

    The goal is to pre-process the building blocks to identify applicable reactants for each reaction template.
    In other words, filter out all building blocks that do not match any reaction template. There is no need to
    keep them, as they cannot act as reactant.
    In a first step, we match all building blocks with each reaction template.
    In a second step, we save all matched building blocks and a collection of Reactions with their available building
    blocks.

    If the file is already present, it will not be recomputed.
    But you can set force to True to bypass any existing intermediate file.


    Args:
        project_root: Path to the project root
        bblocks: Raw building blocks, unfiltered
        force: Whether to bypass any existing file

    Returns:
        - The filtered building blocks
        - The reactions set
    """
    rxn_templates_path = project_root / "data" / "assets" / "reaction-templates" / "hb.txt"
    bblocks_preprocess_path = project_root / "data" / "pre-process" / "building-blocks-rxns"
    bblocks_filtered_path = bblocks_preprocess_path / "bblocks-enamine-us.csv.gz"
    rxn_collection_path = bblocks_preprocess_path / "rxns-hb-enamine-us.json.gz"

    skip_bblocks = should_skip("filtered building blocks", "compute", bblocks_filtered_path, force)
    skip_rxn = should_skip("rections", "compute", rxn_collection_path, force)

    if skip_bblocks and skip_rxn:
        bblocks = BuildingBlockFileHandler().load(str(bblocks_filtered_path))
        rxn_collection = ReactionSet().load(str(rxn_collection_path))
        return bblocks, rxn_collection
    # If either of the files if present, we need to clean them such that they can be recomputed correctly
    elif skip_bblocks:
        safe_remove(bblocks_filtered_path)
    elif skip_rxn:
        safe_remove(rxn_collection_path)

    # Load assets
    rxn_templates = ReactionTemplateFileHandler().load(str(rxn_templates_path))

    bbf = BuildingBlockFilter(
        building_blocks=bblocks,
        rxn_templates=rxn_templates,
        verbose=True
    )
    # Time intensive task...
    bbf.filter()

    # Save on disk
    bblocks = bbf.building_blocks_filtered
    BuildingBlockFileHandler().save(str(bblocks_filtered_path), bblocks)

    # Save collection of reactions which have "available reactants" set (for convenience)
    rxn_collection = ReactionSet(bbf.rxns)
    rxn_collection.save(str(rxn_collection_path))
    return bblocks, rxn_collection


def compute_embeddings(project_root: Path, bblocks: list[smile], cpu_cores: int, force=False) -> MolEmbedder:
    """
    This function is equivalent to script/02-compute-embeddings.py

    We use the embedding space for the building blocks a lot.
    Hence, we pre-compute and store the building blocks.

    If the file is already present, it will not be recomputed.
    But you can set force to True to bypass any existing intermediate file.

    Args:
        project_root: Path to the project root
        bblocks: Raw building blocks, unfiltered
        cpu_cores: Number of cpu cores to use for computation
        force: Whether to bypass any existing file

    Returns:
        The loader/computed molecule embedder
    """
    mol_embedder_path = project_root / "data" / "pre-process" / "embeddings" / "hb-enamine-embeddings.npy"

    mol_embedder = MolEmbedder(processes=cpu_cores)

    if should_skip("embeddings", "compute", mol_embedder_path, force):
        return mol_embedder.load_precomputed(str(mol_embedder_path))

    func = partial(mol_fp, _radius=2, _nBits=256)
    mol_embedder = mol_embedder.compute_embeddings(func, bblocks)
    mol_embedder.save_precomputed(mol_embedder_path)
    return mol_embedder
