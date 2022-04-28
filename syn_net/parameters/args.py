"""
Define default parameters and paths to data.
"""
from pathlib import Path

HOME        = str(Path.home())
SYNNET_PATH = f"{HOME}/SynNet/"

paths = {
	'bb_emb'          : f'{SYNNET_PATH}tests/data/ref/building_blocks_emb.npy',
    'reaction_file'   : f'{SYNNET_PATH}tests/data/ref/rxns_hb.json.gz',
    'building_blocks' : f'{SYNNET_PATH}tests/data/building_blocks_matched.csv.gz',
    'to_act'          : f'{SYNNET_PATH}tests/data/ref/act.ckpt',
    'to_rt1'          : f'{SYNNET_PATH}tests/data/ref/rt1.ckpt',
    'to_rxn'          : f'{SYNNET_PATH}tests/data/ref/rxn.ckpt',
    'to_rt2'          : f'{SYNNET_PATH}tests/data/ref/rt2.ckpt',
	'states'          : f'{SYNNET_PATH}tests/data/ref/',
	'steps'           : f'{SYNNET_PATH}tests/data/ref/',
}

parameters = {
    'rxn_template'  : 'unittest',
	'out_dim'       : 300,
    'nbits'         : 4096,
    'featurize'     : 'fp',
    'ncpu'          : 16,
}


parameters_wenhao = {
    'rxn_template'  : 'hb',
    'out_dim'       : 256,
    'nbits'         : 4096,
    'featurize'     : 'fp',
    'ncpu'          : 16,
}

paths_wenhao = {
	'bb_emb'          : '/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_fp_256.npy',
    'bb_emb_gin'      : '/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_gin.npy',
    'bb_emb_gin'      : '/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_gin.npy',
    'bb_emb_fp_4096'  : '/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_fp_4096.npy',
    'bb_emb_fp_256'   : '/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_fp_256.npy',
    'bb_emb_rdkit2d'  : '/pool001/whgao/data/synth_net/st_hb/enamine_us_emb_rdkit2d.npy',
    'reaction_file'   : f'/pool001/whgao/data/synth_net/st_{parameters_wenhao["rxn_template"]}/reactions_{parameters_wenhao["rxn_template"]}.json.gz',
    'building_blocks' : f'/pool001/whgao/data/synth_net/st_{parameters_wenhao["rxn_template"]}/enamine_us_matched.csv.gz',
    'to_act'          : f'/home/whgao/synth_net/synth_net/params/hb_fp_2_4096_256/act.ckpt',
    'to_rt1'          : f'/home/whgao/synth_net/synth_net/params/hb_fp_2_4096_256/rt1.ckpt',
    'to_rxn'          : f'/home/whgao/synth_net/synth_net/params/hb_fp_2_4096_256/rxn.ckpt',
    'to_rt2'          : f'/home/whgao/synth_net/synth_net/params/hb_fp_2_4096_256/rt2.ckpt',
	'states'          : f'/home/rociomer/data/synth_net/pis_fp/',
	'steps'           : f'/home/rociomer/data/synth_net/pis_fp/',
}