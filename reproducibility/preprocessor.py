from pathlib import Path

from synnet.data_generation.preprocessing import (
    BuildingBlockFileHandler,
    BuildingBlockFilter,
    ReactionTemplateFileHandler,
)
from downloader import should_skip, safe_remove
from synnet.utils.data_utils import ReactionSet
from synnet.encoding.fingerprints import mol_fp
from synnet.MolEmbedder import MolEmbedder

from functools import partial


def filter_bblocks(project_root: Path, bblocks_filtered, cpu_cores, force=False):
    rxn_templates_path = project_root / "data" / "assets" / "reaction-templates" / "hb.txt"
    bblocks_preprocess_path = project_root / "data" / "pre-process" / "building-blocks-rxns"
    bblocks_filtered_path = bblocks_preprocess_path / "bblocks-enamine-us.csv.gz"
    rxn_collection_path = bblocks_preprocess_path / "rxns-hb-enamine-us.json.gz"

    skip_bblocks = should_skip("filtered building blocks", "compute", bblocks_filtered_path, force)
    skip_rxn = should_skip("rections", "compute", rxn_collection_path, force)

    if skip_bblocks and skip_rxn:
        bblocks_filtered = BuildingBlockFileHandler().load(str(bblocks_filtered_path))
        rxn_collection = ReactionSet().load(str(rxn_collection_path))
        return bblocks_filtered, rxn_collection
    # If either of the files if present, we need to clean them such that they can be recomputed correctly
    elif skip_bblocks:
        safe_remove(bblocks_filtered_path)
    elif skip_rxn:
        safe_remove(rxn_collection_path)

    # Load assets
    rxn_templates = ReactionTemplateFileHandler().load(str(rxn_templates_path))

    bbf = BuildingBlockFilter(
        building_blocks=bblocks_filtered,
        rxn_templates=rxn_templates,
        verbose=True,
        processes=cpu_cores,
    )
    # Time intensive task...
    bbf.filter()

    # Save on disk
    bblocks_filtered = bbf.building_blocks_filtered
    BuildingBlockFileHandler().save(str(bblocks_filtered_path), bblocks_filtered)

    # Save collection of reactions which have "available reactants" set (for convenience)
    rxn_collection = ReactionSet(bbf.rxns)
    rxn_collection.save(str(rxn_collection_path))
    return bblocks_filtered, rxn_collection


def compute_embeddings(project_root, bblocks, cpu_cores, force=False):
    molembedder_path = project_root / "data" / "pre-process" / "embeddings" / "hb-enamine-embeddings.npy"

    molembedder = MolEmbedder(processes=cpu_cores)

    if should_skip("embeddings", "compute", molembedder_path, force):
        return molembedder.load_precomputed(molembedder_path)

    func = partial(mol_fp, _radius=2, _nBits=256)
    molembedder = molembedder.compute_embeddings(func, bblocks)
    molembedder.save_precomputed(molembedder_path)
    return molembedder
