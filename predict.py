import torch, tqdm, os, wandb, json, time
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from alphaflow.data.data_modules import collate_fn
from alphaflow.model.wrapper import AlphaFoldWrapper
from alphaflow.utils.tensor_utils import tensor_tree_map
import alphaflow.utils.protein as protein
from alphaflow.data.inference import AlphaFoldCSVDataset, CSVDataset
from collections import defaultdict

from openfold.utils.import_weights import import_jax_weights_
from alphaflow.config import model_config

from alphaflow.utils.logging import get_logger
from alphaflow.model.new_model import AlphaSAXS

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='splits/transporters_only.csv')
parser.add_argument('--templates_dir', type=str, default=None)
parser.add_argument('--msa_dir', type=str, default='./alignment_dir')
parser.add_argument('--saxs_dir', type=str, default='./')
parser.add_argument('--mode', choices=['alphafold', 'esmfold'], default='alphafold')
parser.add_argument('--samples', type=int, default=10)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--outpdb', type=str, default='./outpdb/default')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--original_weights', action='store_true')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--resample', action='store_true')
parser.add_argument('--tmax', type=float, default=1.0)
parser.add_argument('--no_diffusion', action='store_true', default=False)
parser.add_argument('--self_cond', action='store_true', default=False)
parser.add_argument('--noisy_first', action='store_true', default=False)
parser.add_argument('--runtime_json', type=str, default=None)
parser.add_argument('--no_overwrite', action='store_true', default=False)
args = parser.parse_args()


logger = get_logger(__name__)
torch.set_float32_matmul_precision("medium")

config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
) 
schedule = np.linspace(args.tmax, 0, args.steps+1)
if args.tmax != 1.0:
    schedule = np.array([1.0] + list(schedule))
loss_cfg = config.loss
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

if args.subsample: # https://elifesciences.org/articles/75751#s3
    data_cfg.predict.max_msa_clusters = args.subsample // 2
    data_cfg.predict.max_extra_msa = args.subsample

@torch.no_grad()
def main():

    ## Args.data_dir should contain the fasta files

    valset = AlphaFoldCSVDataset(
        config = data_cfg,
        path = args.input_csv,
        saxs_dir = args.saxs_dir,
        msa_dir = args.msa_dir,
    )  
    # valset[0]
    logger.info("Loading the model")
    model_class = AlphaSAXS

    if args.weights:
        ckpt = torch.load(args.weights, map_location='cpu')
        model = model_class(**ckpt['hyper_parameters'], training=False)
        model.model.load_state_dict(ckpt['params'], strict=False)
        model = model.cuda()
        
    
    elif args.original_weights:
        model = model_class(config, None, training=False)
        import_jax_weights_(model.model, 'params_model_1.npz', version='model_3')
        model = model.cuda()
        
    else:
        model = model_class.load_from_checkpoint(args.ckpt, map_location='cpu')
        model.load_ema_weights()
        model = model.cuda()
    model.eval()
    
    logger.info("Model has been loaded")
    
    results = defaultdict(list)
    os.makedirs(args.outpdb, exist_ok=True)
    runtime = defaultdict(list)


    for i, item in enumerate(valset):
        for k in item.keys():
            if not torch.allclose(item[k],valset[i][k]):
                print("Different {} from the very start {}".format(k,i))
        if not torch.allclose(item['msa_feat'],valset[i]['msa_feat']):
            print("Different MSA from the very start {}".format(i))
        if args.pdb_id and item['name'] not in args.pdb_id:
            continue
        if args.no_overwrite and os.path.exists(f'{args.outpdb}/{item["name"]}.pdb'):
            continue
        result = []
        if not torch.allclose(item['msa_feat'],valset[i]['msa_feat']):
            print("Different MSA from the start") 
        for j in tqdm.trange(args.samples):
            if not torch.allclose(item['msa_feat'],valset[i]['msa_feat']):
                print("Different MSA at the {} itteration".format(j)) 
            if args.subsample or args.resample:
                # Get item function                
                item = valset[i] # resample MSA
            start = time.time()
            batch = collate_fn([item])
            batch = tensor_tree_map(lambda x: x.cuda(), batch)
            prots = model.inference(batch, as_protein=True, noisy_first=args.noisy_first,
                        no_diffusion=args.no_diffusion, schedule=schedule, self_cond=args.self_cond)
            runtime[item['name']].append(time.time() - start)
            result.append(prots[-1])
            


        with open(f'{args.outpdb}/{item["name"]}.pdb', 'w') as f:
            f.write(protein.prots_to_pdb(result))

    if args.runtime_json:
        with open(args.runtime_json, 'w') as f:
            f.write(json.dumps(dict(runtime)))
if __name__ == "__main__":
    main()
