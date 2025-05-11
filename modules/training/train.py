import argparse
import os
import os.path as osp
import time
import sys
import glob
import tqdm
import wandb  # Import wandb

def parse_arguments():
    parser = argparse.ArgumentParser(description="XFeat training script.")

    parser.add_argument('--megadepth_paths', type=str, nargs='+', required=True,
                        help='List of paths to the MegaDepth_v1 dataset parts. Example: "/path/part1 /path/part2 /path/part3 /path/part4"')
    parser.add_argument('--megadepth_metadata_path', type=str, required=True,
                        help='Path to the MegaDepth dataset metadata directory (contains the npz files).')
    parser.add_argument('--synthetic_root_path', type=str, default='/homeLocal/guipotje/sshfs/datasets/coco_20k',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--training_type', type=str, default='xfeat_default',
                        choices=['xfeat_default', 'xfeat_synthetic', 'xfeat_megadepth'],
                        help='Training scheme. xfeat_default uses both megadepth & synthetic warps.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training. Default is 10.')
    parser.add_argument('--n_steps', type=int, default=160_000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=500,
                        help='Save checkpoints every N steps. Default is 500.')
    # Add arguments for resuming training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to the checkpoint to resume training from.')
    parser.add_argument('--start_step', type=int, default=0,
                        help='Step to start training from when resuming. Default is 0.')
    parser.add_argument('--wandb_project', type=str, default='xfeat-training',
                        help='Weights & Biases project name. Default is "xfeat-training".')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity (username or team name). Default is None (uses default entity).')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name. Default is None (auto-generated name).')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='Weights & Biases run ID to resume. Default is None (creates new run).')
    parser.add_argument('--skip_wandb', action='store_true',
                        help='If set, skip using Weights & Biases for logging.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args

args = parse_arguments()

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from modules.model import *
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *

from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth import megadepth_warper
from torch.utils.data import Dataset, DataLoader

class Trainer():
    """
    Class for training XFeat with default params as described in the paper.
    We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
    Modified to handle multiple MegaDepth dataset parts using a scene ID mapping system.
    """

    def __init__(self, megadepth_paths, 
                       megadepth_metadata_path,
                       synthetic_root_path, 
                       ckpt_save_path, 
                       model_name = 'xfeat_default',
                       batch_size = 10, n_steps = 160_000, lr= 3e-4, gamma_steplr=0.5, 
                       training_res = (800, 608), device_num="0", dry_run = False,
                       save_ckpt_every = 500, use_wandb = True, wandb_project = 'xfeat-training',
                       wandb_entity = None, wandb_run_name = None, wandb_run_id = None,
                       resume_from_checkpoint = None, start_step = 0):

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev)
        
        # Track the starting step for resuming training
        self.start_step = start_step
        
        # Load checkpoint if resuming training
        if resume_from_checkpoint is not None:
            print(f"Loading checkpoint from {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location=self.dev)
            self.net.load_state_dict(checkpoint)
            print(f"Successfully loaded checkpoint. Resuming from step {start_step}")
        
        # Setup Weights & Biases
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb_config = {
                "model_name": model_name,
                "batch_size": batch_size,
                "n_steps": n_steps,
                "learning_rate": lr,
                "gamma_steplr": gamma_steplr,
                "training_resolution": training_res,
                "save_checkpoint_every": save_ckpt_every,
                "megadepth_paths_count": len(megadepth_paths),
                "device": device_num,
                "resume_training": resume_from_checkpoint is not None,
                "start_step": start_step
            }
            
            # Resume wandb run if ID is provided, otherwise create new run
            resume = "must" if wandb_run_id else None
            
            self.run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name or f"{model_name}_{time.strftime('%Y_%m_%d-%H_%M_%S')}",
                id=wandb_run_id,
                resume=resume,
                config=wandb_config
            )
            
            # Log model architecture
            wandb.watch(self.net, log="all", log_freq=100)

        # Setup optimizer 
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr)
        
        # Set initial_lr in optimizer param groups (needed for resuming)
        for param_group in self.opt.param_groups:
            param_group['initial_lr'] = lr
        
        # Adjust scheduler for resuming training
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, 
            step_size=30_000, 
            gamma=gamma_steplr,
            last_epoch=start_step-1 if start_step > 0 else -1  # Tell the scheduler where we are
        )

        ##################### Synthetic COCO INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_synthetic'):
            self.augmentor = AugmentationPipe(
                                        img_dir = synthetic_root_path,
                                        device = self.dev, load_dataset = True,
                                        batch_size = int(self.batch_size * 0.4 if model_name=='xfeat_default' else batch_size),
                                        out_resolution = training_res, 
                                        warp_resolution = training_res,
                                        sides_crop = 0.1,
                                        max_num_imgs = 3_000,
                                        num_test_imgs = 5,
                                        photometric = True,
                                        geometric = True,
                                        reload_step = 4_000
                                        )
        else:
            self.augmentor = None
        ##################### Synthetic COCO END #######################


        ##################### MEGADEPTH INIT ##########################
        if model_name in ('xfeat_default', 'xfeat_megadepth'):
            # Create a mapping of scene ID ranges to dataset paths
            self.root_dirs = self.create_scene_id_mapping(megadepth_paths)

            print(f"Root directory{self.root_dirs}")
            
            # Initialize the MegaDepth data loader with scene ID mapping
            self.setup_megadepth_loader(megadepth_metadata_path, model_name)
        else:
            self.root_dirs = None
            self.data_iter = None
        ##################### MEGADEPTH INIT END #######################

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name
        
    def create_scene_id_mapping(self, megadepth_paths):
        """Create a mapping from scene ID ranges to root directories with directory verification"""
        # Standard MegaDepth dataset ranges
        scene_ranges = [
            (0, 47),      # MegaDepth_p1
            (48, 159),    # MegaDepth_p2
            (160, 326),   # MegaDepth_p3
            (327, 5018)   # MegaDepth_p4
        ]
        
        root_dirs = {}
        for i, path in enumerate(megadepth_paths):
            if i < len(scene_ranges):
                root_dirs[scene_ranges[i]] = path
                
                # Verify some scene directories exist in this path
                start_id, end_id = scene_ranges[i]
                found_dirs = []
                for scene_id in range(start_id, min(start_id + 10, end_id + 1)):
                    scene_dir = f"{scene_id:04d}"
                    if osp.exists(osp.join(path, scene_dir)):
                        found_dirs.append(scene_id)
                
                print(f"Path {path} (range {start_id:04d}-{end_id:04d}) contains scenes: {found_dirs}")
        
        return root_dirs
        
    def setup_megadepth_loader(self, metadata_path, model_name):
        """Set up the MegaDepth data loader with scene ID to directory mapping"""
        print(f"Setting up MegaDepth data loader with scene ID mapping")
        
        TRAIN_BASE_PATH = metadata_path
        TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

        npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
        
        # Pass the root_dirs mapping to the MegaDepthDataset
        datasets = []
        for path in tqdm.tqdm(npz_paths, desc=f"[MegaDepth] Loading metadata"):
            try:
                # Create dataset for this NPZ file
                dataset = MegaDepthDataset(root_dirs=self.root_dirs, npz_path=path)
                datasets.append(dataset)
            except Exception as e:
                print(f"Error loading dataset from {path}: {e}")
                continue
        
        data = torch.utils.data.ConcatDataset(datasets)
        
        print(f"Megadepth metadata loading finished. Total samples: {len(data)}")
        print(f"Dataset type: {type(data)}")
        print(f"Number of individual datasets: {len(data.datasets)}")
        
        # Add dataset integrity verification
        print("Verifying dataset integrity...")
        success_count = 0
        error_count = 0
        
        # Sample a few items to verify integrity
        sample_indices = np.random.choice(len(data), min(5, len(data)), replace=False)
        for i in sample_indices:
            try:
                # Just try to access the item to verify paths
                _ = data[i]
                success_count += 1
            except Exception as e:
                print(f"Error accessing item {i}: {e}")
                error_count += 1
        
        print(f"Dataset verification complete. Successes: {success_count}, Errors: {error_count}")
        
        # Create data loader
        self.data_loader = DataLoader(
            data, 
            batch_size=int(self.batch_size * 0.6 if model_name=='xfeat_default' else self.batch_size),
            shuffle=True
        )
        self.data_iter = iter(self.data_loader)
        
    def save_checkpoint(self, step):
        """Save model checkpoint both locally and to wandb"""
        checkpoint_path = f"{self.ckpt_save_path}/{self.model_name}_{step}.pth"
        torch.save(self.net.state_dict(), checkpoint_path)
        
        if self.use_wandb:
            wandb.save(checkpoint_path)
            
        print(f"Saved checkpoint at step {step}")

    def train(self):
        self.net.train()

        difficulty = 0.10

        p1s, p2s, H1, H2 = None, None, None, None
        d = None

        if self.augmentor is not None:
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)
        
        if self.data_iter is not None:
            d = next(self.data_iter)

        # Starting from the specified step if resuming
        total_steps = self.start_step + self.steps
        
        with tqdm.tqdm(total=self.steps, initial=self.start_step) as pbar:
            for i in range(self.start_step, total_steps):
                if not self.dry_run:
                    if self.data_iter is not None:
                        try:
                            # Get the next MegaDepth batch
                            d = next(self.data_iter)
                        except StopIteration:
                            print("End of dataset. Reinitializing iterator.")
                            # If StopIteration is raised, create a new iterator
                            self.data_iter = iter(self.data_loader)
                            d = next(self.data_iter)

                    if self.augmentor is not None:
                        # Grab synthetic data
                        p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

                if d is not None:
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)
                
                    p1, p2 = d['image0'], d['image1']
                    positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)

                if self.augmentor is not None:
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _, positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)

                # Join megadepth & synthetic data
                with torch.inference_mode():
                    # RGB -> GRAY
                    if d is not None:
                        p1 = p1.mean(1, keepdim=True)
                        p2 = p2.mean(1, keepdim=True)
                    if self.augmentor is not None:
                        p1s = p1s.mean(1, keepdim=True)
                        p2s = p2s.mean(1, keepdim=True)

                    # Cat two batches
                    if self.model_name in ('xfeat_default'):
                        p1 = torch.cat([p1s, p1], dim=0)
                        p2 = torch.cat([p2s, p2], dim=0)
                        positives_c = positives_s_coarse + positives_md_coarse
                    elif self.model_name in ('xfeat_synthetic'):
                        p1 = p1s ; p2 = p2s
                        positives_c = positives_s_coarse
                    else:
                        positives_c = positives_md_coarse

                # Check if batch is corrupted with too few correspondences
                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                # Forward pass
                feats1, kpts1, hmap1 = self.net(p1)
                feats2, kpts2, hmap2 = self.net(p2)

                loss_items = []

                for b in range(len(positives_c)):
                    # Get positive correspondencies
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                    # Grab features at corresponding idxs
                    m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                    m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)

                    # grab heatmaps at corresponding idxs
                    h1 = hmap1[b, 0, pts1[:,1].long(), pts1[:,0].long()]
                    h2 = hmap2[b, 0, pts2[:,1].long(), pts2[:,0].long()]
                    coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                    # Compute losses
                    loss_ds, conf = dual_softmax_loss(m1, m2)
                    loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)

                    loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1[b], p1[b])
                    loss_kp_pos2, acc_pos2 = alike_distill_loss(kpts2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2)*2.0
                    acc_pos = (acc_pos1 + acc_pos2)/2

                    loss_kp = keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(loss_coords.unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    loss_items.append(loss_kp_pos.unsqueeze(0))

                    if b == 0:
                        acc_coarse_0 = check_accuracy(m1, m2)

                acc_coarse = check_accuracy(m1, m2)

                nb_coarse = len(m1)
                loss = torch.cat(loss_items, -1).mean()
                loss_coarse = loss_ds.item()
                loss_coord = loss_coords.item()
                loss_kp_pos = loss_kp_pos.item()
                loss_l1 = loss_kp.item()

                # Compute Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i+1) % self.save_ckpt_every == 0:
                    self.save_checkpoint(i+1)

                # Calculate learning rate
                current_lr = self.scheduler.get_last_lr()[0]

                # Get current scene ID if available in the batch
                scene_id = d.get('scene_id', [-1])[0] if d is not None else -1

                pbar.set_description(
                    f'Loss: {loss.item():.4f} acc_c0 {acc_coarse_0:.3f} acc_c1 {acc_coarse:.3f} '
                    f'acc_f: {acc_coords:.3f} loss_c: {loss_coarse:.3f} loss_f: {loss_coord:.3f} '
                    f'loss_kp: {loss_l1:.3f} #matches_c: {nb_coarse:d} loss_kp_pos: {loss_kp_pos:.3f} '
                    f'acc_kp_pos: {acc_pos:.3f} | Scene ID: {scene_id}'
                )
                pbar.update(1)

                # Log metrics to TensorBoard
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
                self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
                self.writer.add_scalar('Accuracy/fine_mdepth', acc_coords, i)
                self.writer.add_scalar('Accuracy/kp_position', acc_pos, i)
                self.writer.add_scalar('Loss/coarse', loss_coarse, i)
                self.writer.add_scalar('Loss/fine', loss_coord, i)
                self.writer.add_scalar('Loss/reliability', loss_l1, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_kp_pos, i)
                self.writer.add_scalar('Count/matches_coarse', nb_coarse, i)
                self.writer.add_scalar('Dataset/scene_id', scene_id, i)
                self.writer.add_scalar('Training/learning_rate', current_lr, i)
                
                # Log metrics to Weights & Biases
                if self.use_wandb:
                    wandb_log = {
                        'train/loss': loss.item(),
                        'train/loss_coarse': loss_coarse,
                        'train/loss_fine': loss_coord,
                        'train/loss_keypoint': loss_l1,
                        'train/loss_keypoint_pos': loss_kp_pos,
                        'train/accuracy_coarse_synth': acc_coarse_0,
                        'train/accuracy_coarse_mdepth': acc_coarse,
                        'train/accuracy_fine_mdepth': acc_coords,
                        'train/accuracy_keypoint_pos': acc_pos,
                        'train/matches_coarse_count': nb_coarse,
                        'train/learning_rate': current_lr,
                        'dataset/scene_id': scene_id
                    }
                    wandb.log(wandb_log, step=i)

        # Save final model checkpoint
        self.save_checkpoint(total_steps)
        
        # Close wandb run
        if self.use_wandb:
            wandb.finish()


if __name__ == '__main__':
    # Initialize Weights & Biases if not skipped
    if not args.skip_wandb:
        # Login to Weights & Biases (only needed once)
        try:
            import wandb
            # Quietly log in or use cached credentials
            wandb.login(anonymous="allow")
            print("Successfully logged in to Weights & Biases")
        except Exception as e:
            print(f"Failed to log in to Weights & Biases: {e}")
            print("Setting skip_wandb to True")
            args.skip_wandb = True
    
    trainer = Trainer(
        megadepth_paths=args.megadepth_paths,
        megadepth_metadata_path=args.megadepth_metadata_path,
        synthetic_root_path=args.synthetic_root_path, 
        ckpt_save_path=args.ckpt_save_path,
        model_name=args.training_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every,
        use_wandb=not args.skip_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_run_id=args.wandb_run_id,
        resume_from_checkpoint=args.resume_from_checkpoint,
        start_step=args.start_step
    )

    # The most fun part
    trainer.train()