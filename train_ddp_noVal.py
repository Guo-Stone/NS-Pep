import wandb

import os
import pdb
import shutil
import argparse
import torch
import torch.cuda.amp as amp
import torch.distributed as distrib
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import math

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from pepflow.utils.vc import get_version, has_changes
from pepflow.utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, \
    current_milli_time
from pepflow.utils.data import PaddingCollate
from pepflow.utils.train import ScalarMetricAccumulator, count_parameters, get_optimizer, get_scheduler, log_losses, \
    recursive_to, sum_weighted_losses

from models_con.pep_dataloader import PepDataset
# from models_con.flow_model import FlowModel
from models_con.flow_model import FlowModel, FlowModel_tao, FlowModel_SEP
import models_con.flow_model as flow_model


class WeightedDistributedSampler(Sampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True, num_samples=None):
        """
        A sampler that combines `WeightedRandomSampler` and `DistributedSampler`.

        Args:
            dataset: Dataset object.
            weights: A tensor of weights for each data point.
            num_replicas: Number of processes participating in distributed training.
            rank: Rank of the current process within num_replicas.
            replacement: Whether to sample with replacement.
            num_samples: Total number of samples to draw.
        """
        self.dataset = dataset
        self.weights = weights
        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()
        self.replacement = replacement

        if num_samples is None:
            self.num_samples = int(math.ceil(len(dataset) / self.num_replicas))
        else:
            self.num_samples = num_samples

        # Ensure dataset is evenly divisible
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # Compute the start and end indices for this rank
        start_idx = self.rank * self.num_samples
        end_idx = start_idx + self.num_samples

        # Adjust weights for this rank's shard
        indices = torch.arange(len(self.dataset))
        shuffled_indices = indices[start_idx:end_idx]

        # Apply weights to the local shard
        local_weights = self.weights[shuffled_indices]

        # Sample using the local weights
        sampled_indices = torch.multinomial(local_weights, self.num_samples, self.replacement)

        return iter(shuffled_indices[sampled_indices].tolist())

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/learn_angle_tao.yaml')
    parser.add_argument('--logdir', type=str, default="./logs")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank. Necessary for using the torch.distributed.launch utility.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--name', type=str, default='pepflow-NS3')
    parser.add_argument('--weight_sample', action='store_true', default=False)
    args = parser.parse_args()

    # local_rank = args.local_rank
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Version control
    # branch, version = get_version()
    branch, version = "gt", "522"
    version_short = '%s-%s' % (branch, version[:7])
    # version_short = 'g-t'
    # if has_changes() and not args.debug:
    #     c = input('Start training anyway? (y/n) ')
    #     if c != 'y':
    #         exit()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed + local_rank * 100)

    # Logging
    if args.debug or local_rank > 0:
        run = wandb.init(project=args.name, config=config, name='%s[%s]' % (config_name, args.tag), mode="disabled")
        logger = get_logger('train', None, local_rank)
        writer = BlackHole()
    else:
        pass  # Set WANDB_API_KEY in your shell environment (do not hard-code it)
        run = wandb.init(project=args.name, config=config, name='%s[%s]' % (config_name, args.tag))
        # if args.resume:
        #     log_dir = os.path.dirname(os.path.dirname(args.resume))
        # else:
        log_dir = get_new_log_dir(args.logdir, prefix='%s[%s]' % (config_name, version_short), tag=args.tag)
        with open(os.path.join(log_dir, 'commit.txt'), 'w') as f:
            f.write(branch + '\n')
            f.write(version + '\n')
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = BlackHole()
        # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        # tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Set up DDP
    logger.info('Initializing DDP...')
    distrib.init_process_group(backend="nccl")

    # Data
    logger.info('Loading datasets...')
    train_dataset = PepDataset(structure_dir=config.dataset.train.structure_dir,
                               dataset_dir=config.dataset.train.dataset_dir,
                               name=config.dataset.train.name, transform=None,
                               reset=config.dataset.train.reset,
                               tao_repr=config.dataset.train.tao_repr,
                               split=config.dataset.split)
    # val_dataset = PepDataset(structure_dir=config.dataset.val.structure_dir,
    #                          dataset_dir=config.dataset.val.dataset_dir,
    #                          name=config.dataset.val.name, transform=None,
    #                          reset=config.dataset.val.reset,
    #                          tao_repr=False,
    #                          split=config.dataset.split)

    if args.weight_sample:
        logger.info('Sampling according to weights...')
        weights = torch.tensor([item['weight'] for item in train_dataset])
        train_sampler = WeightedDistributedSampler(train_dataset, weights, replacement=True)
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(),
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    train_iterator = inf_iterator(train_loader)

    # val_sampler = DistributedSampler(val_dataset, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(),
    #                         sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    # val_iterator = inf_iterator(val_loader)
    logger.info('Train %d' % (len(train_dataset)))

    # Model
    logger.info('Building model...')
    if config.dataset.train.tao_repr:
        # model = DDP(FlowModel_tao(config.model).to(local_rank),
        #             device_ids=[local_rank], find_unused_parameters=True)
        # model = DDP(flow_model.FlowModel_NS3_SCAux(config.model).to(local_rank), device_ids=[local_rank])
        model = DDP(flow_model.NS_Pep(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_BLV(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_seesaw(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_wo_rBLV(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.FlowModel_NS3_weight_loss(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_dyMEAN_v1(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_dyMEAN_v2(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_wo_GSP_v2(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_wo_IAW(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_RFGM_no_noise(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_BLV_no_noise(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_only_noise(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_tsample(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.NS_Pep_t75_predSC(config.model).to(local_rank), device_ids=[local_rank])

    else:
        # model = DDP(FlowModel(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(FlowModel_SEP(config.model).to(local_rank), device_ids=[local_rank])
        model = DDP(flow_model.FlowModel_NS3(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.FlowModel_NS3_distEdge(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.FlowModel_NS3_BLV(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.FlowModel_NS3_rBLV(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.FlowModel_NS3_rBLV_norm(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.FlowModel_NS3_BLV_acc(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.FlowModel_NS3_seesaw(config.model).to(local_rank), device_ids=[local_rank])
        # model = DDP(flow_model.FlowModel_NS3_transfer(config.model).to(local_rank), device_ids=[local_rank])
    # wandb.watch(model,log='all',log_freq=1)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)

        ckpt = torch.load(args.resume, map_location=f'cuda:{local_rank}')
        pretrained_state_dict = ckpt['model']

        model_state_dict = model.state_dict()

        # Filter out keys that are not in the current model's state dict
        matched_pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if
                                         k in model_state_dict and v.size() == model_state_dict[k].size()}

        # and not k.startswith("module.ga_encoder")}
        # Load the filtered state dict with strict=False to ignore unmatched layers
        model_state_dict.update(matched_pretrained_state_dict)
        """
        # DEBUG
        # unmatched_block = []
        unmatched_block = ['module.node_embedder.aatype_embed.weight', 'module.node_embedder.mlp.0.weight',
                           'module.edge_embedder.aa_pair_embed.weight',
                           'module.edge_embedder.aapair_to_distcoef.weight',
                           'module.edge_embedder.distance_embed.0.weight',
                           'module.ga_encoder.current_seq_embedder.weight',
                           'module.ga_encoder.seq_net.4.weight', 'module.ga_encoder.seq_net.4.bias']
        # for k, v in pretrained_state_dict.items():
        #     if k in model_state_dict and v.size() != model_state_dict[k].size():
        #         unmatched_block.append(k)
        unmatched_linear_block = ['module.node_embedder.aatype_embed.weight',  # +2
                                  'module.ga_encoder.current_seq_embedder.weight',  # +2
                                  'module.ga_encoder.seq_net.4.weight',  # +0
                                  'module.ga_encoder.seq_net.4.bias']  # +0
        unmatched_complex_block = ['module.node_embedder.mlp.0.weight',  # [256, 1442]
                                   'module.edge_embedder.aa_pair_embed.weight',  # 625
                                   'module.edge_embedder.aapair_to_distcoef.weight',  # 625
                                   'module.edge_embedder.distance_embed.0.weight', ]  # [64, 289]
        for layer in unmatched_linear_block:
            channel_num = pretrained_state_dict[layer].shape[0]
            model_state_dict[layer][:channel_num] = pretrained_state_dict[layer]

        # Deal with 'module.node_embedder.mlp.0.weight'. * feat_dim = 128
        feat_dim = 128
        old_max_num_atoms = 15
        new_max_num_atoms = 17
        old_max_aa_types = 22
        new_max_aa_types = 25
        start_idx = feat_dim
        new_end_idx = feat_dim + (new_max_aa_types * new_max_num_atoms * 3)
        tgt_param =\
            model_state_dict['module.node_embedder.mlp.0.weight'][:, start_idx: new_end_idx]\
            .reshape(-1, new_max_aa_types, new_max_num_atoms, 3).clone()
        old_end_idx = feat_dim + (old_max_aa_types * old_max_num_atoms * 3)

        tgt_param[:, :old_max_aa_types, :old_max_num_atoms] = pretrained_state_dict['module.node_embedder.mlp.0.weight'][:, start_idx: old_end_idx]\
            .reshape(-1, old_max_aa_types, old_max_num_atoms, 3)
        model_state_dict['module.node_embedder.mlp.0.weight'][:, start_idx: new_end_idx] = tgt_param.reshape(-1, new_max_aa_types * new_max_num_atoms * 3)

        # Deal with 'module.edge_embedder.aa_pair_embed.weight'
        tgt_param = model_state_dict['module.edge_embedder.aa_pair_embed.weight'].reshape(new_max_aa_types, new_max_aa_types, -1).clone()
        tgt_param[:old_max_aa_types, :old_max_aa_types] = pretrained_state_dict['module.edge_embedder.aa_pair_embed.weight'].reshape(old_max_aa_types, old_max_aa_types, -1)
        model_state_dict['module.edge_embedder.aa_pair_embed.weight'] = tgt_param.reshape(new_max_aa_types*new_max_aa_types, -1)

        # Deal with 'module.edge_embedder.aapair_to_distcoef.weight'
        tgt_param = model_state_dict['module.edge_embedder.aapair_to_distcoef.weight']\
            .reshape(new_max_aa_types, new_max_aa_types, new_max_num_atoms, new_max_num_atoms)
        tgt_param[:old_max_aa_types, :old_max_aa_types, :old_max_num_atoms, :old_max_num_atoms] =\
            pretrained_state_dict['module.edge_embedder.aapair_to_distcoef.weight'] \
                .reshape(old_max_aa_types, old_max_aa_types, old_max_num_atoms, old_max_num_atoms)
        model_state_dict['module.edge_embedder.aapair_to_distcoef.weight'] =\
            tgt_param.reshape(new_max_aa_types*new_max_aa_types, new_max_num_atoms*new_max_num_atoms)

        # Deal with 'module.edge_embedder.distance_embed.0.weight'
        tgt_param = model_state_dict['module.edge_embedder.distance_embed.0.weight']\
            .reshape(-1, new_max_num_atoms, new_max_num_atoms)
        tgt_param[:, :old_max_num_atoms, :old_max_num_atoms] = pretrained_state_dict['module.edge_embedder.distance_embed.0.weight']\
            .reshape(-1, old_max_num_atoms, old_max_num_atoms)
        model_state_dict['module.edge_embedder.distance_embed.0.weight'] = tgt_param.reshape(-1, new_max_num_atoms*new_max_num_atoms)

        # DEBUG
        """
        model.load_state_dict(model_state_dict, strict=False)
        logger.info('Loaded pretrained weights for matching layers.')
        try:
            logger.info('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.info('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
        except ValueError or RuntimeError:
            pass

    def train(it):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(train_iterator), local_rank)
        # # inspect
        # if local_rank == 0:
        #     torch.autograd.set_detect_anomaly(True)
        # Forward pass
        loss_dict = model(batch)  # get loss and metrics
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        time_forward_end = current_milli_time()

        # if torch.isnan(loss):
        #     print('NAN Loss!')
        #     if local_rank == 0:
        #         torch.save({'batch':batch,'loss':loss,'loss_dict':loss_dict,'model': model.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'scheduler': scheduler.state_dict(),
        #                 'iteration': it,},os.path.join(log_dir,'nan.pt'))
        #     loss = torch.tensor(0.,requires_grad=True).to(loss.device)

        loss.backward()

        # rescue for nan grad
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    param.grad[torch.isnan(param.grad)] = 0

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        # Backward
        # if it % config.train.accum_grad ==0:
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        if local_rank == 0:
            scalar_dict = {}
            # scalar_dict.update(metric_dict['scalar'])
            scalar_dict.update({
                'grad': orig_grad_norm,
                'lr': optimizer.param_groups[0]['lr'],
                'time_forward': (time_forward_end - time_start) / 1000,
                'time_backward': (time_backward_end - time_forward_end) / 1000,
            })
            log_losses(loss, loss_dict, scalar_dict, it=it, tag='train', logger=logger)


    try:
        min_val_loss = 1e3
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.save_freq == 0 and local_rank == 0:
                ckpt_path = os.path.join(log_dir, "checkpoints", '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    # 'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
        distrib.destroy_process_group()
    # except RuntimeError:
    #     logger.info('It seems that peers have been terminated.')
    #     distrib.destroy_process_group()
