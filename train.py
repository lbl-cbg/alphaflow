from alphaflow.utils.parsing import parse_train_args


from alphaflow.utils.logging import get_logger

import torch, tqdm, os, wandb
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.import_weights import import_jax_weights_


from alphaflow.config import model_config
from alphaflow.data.data_modules import OpenFoldSingleDataset, OpenFoldBatchCollator
from alphaflow.data.inference import AlphaFoldCSVDataset
from alphaflow.model.new_model import AlphaSAXS

torch.set_float32_matmul_precision("medium")
args = parse_train_args()
logger = get_logger(__name__)


config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
) 

loss_cfg = config.loss
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

def load_clusters(path):
    cluster_size = []
    with open(args.pdb_clusters) as f:
        for line in f:
            names = line.split()
            for name in names:
                cluster_size.append({'name': name, 'cluster_size': len(names)})
    return pd.DataFrame(cluster_size).set_index('name')
    
def main():
    
    if args.wandb:
        wandb.init(
            project="alphaflow",
            name=args.run_name,
            config=args,
            mode='online'
        )

    logger.info("Loading the chains dataframe")
    pdb_chains = pd.read_csv(args.pdb_chains, index_col='name')

    # Define the training and validation datasets
    # The OpenFoldSingleDataset is altered to include SAXS data.
    trainset = OpenFoldSingleDataset(
        data_dir = args.train_data_dir,
        alignment_dir = args.train_msa_dir,
        saxs_dir=args.saxs_dir,
        pdb_chains = pdb_chains,
        config = data_cfg,
        mode = 'train',
        subsample_pos = args.sample_train_confs,
        first_as_template = args.first_as_template,
    )

    # If normal_validate is set, the validation dataset is created using the same obejcts as the training dataset.
    # And the validation process is using the training function
    if args.normal_validate:
        val_pdb_chains = pd.read_csv(args.val_csv, index_col='name')
        valset = OpenFoldSingleDataset(
            data_dir = args.train_data_dir,
            alignment_dir = args.val_msa_dir,
            saxs_dir=args.saxs_dir,
            pdb_chains = val_pdb_chains,
            config = data_cfg,
            mode = 'train',
            subsample_pos = args.sample_val_confs,
            num_confs = args.num_val_confs,
            first_as_template = args.first_as_template,
        )   
    else:
    # Here the validation dataset is created using a diiferent object which is defined in the inference module.
    # This validation process is using the inference function defined in the model.
        valset = AlphaFoldCSVDataset(
            config = data_cfg,
            path = args.val_csv,
            mmcif_dir=args.mmcif_dir,
            saxs_dir=args.saxs_dir,
            msa_dir=args.val_msa_dir,
        )
    
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        collate_fn=OpenFoldBatchCollator(),
        num_workers=args.num_workers,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        collate_fn=OpenFoldBatchCollator(),
        num_workers=args.num_workers,
        shuffle=not args.filter_chains,
    )

    # I believe we should add another ModelCheckpoint for each epoch.
    trainer = pl.Trainer(
        accelerator="gpu",num_nodes=2 ,devices='auto',
        max_epochs=args.epochs,
        limit_train_batches=args.limit_batches or 1.0,
        limit_val_batches=args.limit_batches or 1.0,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        gradient_clip_val=args.grad_clip,
        callbacks=[ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"], 
            save_top_k=1,
            every_n_train_steps=100,
        ), ModelCheckpoint(dirpath=os.environ["MODEL_DIR"], 
                                save_top_k=1,
                                every_n_epochs=1),],
        accumulate_grad_batches=args.accumulate_grad,
        check_val_every_n_epoch=args.val_freq,
        logger=False, 
        profiler="pytorch"
    )

    # Mode is useless for now. It is always set to alphafold. 
    if args.mode == 'alphafold':
        model = AlphaSAXS(config, args)
        logger.info("Loading the model")
        # I think this part will only load the model at the start of the training. And will be overwritten by the checkpoint.
        import_jax_weights_(model.model, 'params_model_1.npz', version='model_3')
        if not args.no_ema:
            model.ema = ExponentialMovingAverage(
                model=model.model, decay=config.ema.decay
                ) # need to initialize EMA this way at the beginning (Why?)
    else:
        raise ValueError("This part is removed for now.")

#    Should we place this part to the import_jax_weights_ function? And set a if statement for it?

#    if args.restore_weights_only:
#        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'], strict=False)
#        args.ckpt = None
#        if not args.no_ema:
#            model.ema = ExponentialMovingAverage(
#                model=model.model, decay=config.ema.decay
#            ) # need to initialize EMA this way at the beginning
    
    if args.validate:
        trainer.validate(model, val_loader, ckpt_path=args.ckpt)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
    
if __name__ == "__main__":
    main()