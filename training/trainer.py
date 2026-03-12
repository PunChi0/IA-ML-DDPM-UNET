import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from models.unet import *
from models.text_encoder import *
from utils.tokenizer import *
from models.diffusion import *

from torch.amp import autocast, GradScaler

class Trainer:
    def __init__(self, config: dict):
        """
        config keys:
            device          : torch.device
            checkpoint_dir  : str
            epochs          : int
            T               : int
            batch_size      : int
            image_size      : int
        """
        self.config = config
        self.device = config["device"]
        self.checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
        self.epochs = config["epochs"]
        self.T = config["T"]
        self.start_epoch = 0
        self.losses = []
        self.scaler = GradScaler()

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        
    def train(self,unet:UNet,optimizer,text_encoder:TextEncoder,train_loader,word2idx,max_len=30,pretrained:str=None,conditionned:bool=False,tag:str=""):
        
        # Load pretrained weight
        if pretrained != None:
            
            self._load_weight(unet, optimizer, text_encoder, pretrained)
        
        # Init some values for the noise
        beta_start = 1e-4
        beta_end = 0.02

        betas = torch.linspace(beta_start, beta_end, self.T).to(self.device)
        alphas = (1 - betas).to(self.device)
        alpha_bars = torch.cumprod(alphas, dim=0).to(self.device)
        
        # Training loop
        for epoch in range(self.start_epoch,self.epochs):
            
            # Init
            epoch_loss = 0.0
            
            # Put the model into train mode
            unet.train()
            if text_encoder is not None : 
                text_encoder.train()
                
            # Tqdm bar
            batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            
            for batch in batch_bar:
                
                # Get the image
                x0 = batch['image'].to(self.device) # (B, 3, 32, 32)
                B = x0.shape[0]
                
                # Text embedding part
                text_emb = None
                if text_encoder is not None:
                    # Get the text
                    texts = batch['text']

                    # Tokenized the text
                    tokens = torch.tensor([encode(t,max_len,word2idx) for t in texts], dtype=torch.long).to(self.device) # (B, seq_len)
                    padding_mask = (tokens == word2idx['<PAD>']).to(self.device) # (B, seq_len)
                    text_emb = text_encoder(tokens, padding_mask) # (B, seq_len, 128)
                    
                # Noise
                timesteps = torch.randint(0, self.T, (B,), device=self.device)
                x_t, noise = add_noise(x0, timesteps,alpha_bars) # (B, 3, 32, 32)
                
                # Forward + loss
                with autocast('cuda'):
                    noise_pred = unet(x_t, timesteps, text_emb)
                    loss = F.mse_loss(noise_pred, noise)

                # Backprop
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                # Update some values
                epoch_loss += loss.item()
                batch_bar.set_postfix(loss=f"{loss.item():.6f}")

            avg_loss = epoch_loss / len(train_loader)
            self.losses.append(avg_loss)
            
            if epoch % 10 == 0:
                self._save(epoch,unet,optimizer,text_encoder,avg_loss,self.losses,tag)
                print(f"Epoch {epoch:4d} / {self.epochs} | Avg Loss : {avg_loss:.6f} | Saved")
                
    def _save(self, epoch:int, unet:UNet, optimizer, text_encoder:TextEncoder, avg_loss:float, losses, tag: str):
        """Save a checkpoint every 10 epochs."""
        if epoch % 10 != 0:
            return

        raw_model = unet.module if isinstance(unet, nn.DataParallel) else unet
        payload = {
            "epoch" : epoch,
            "unet_state" : raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss" : avg_loss,
            "losses" : losses,
        }

        if text_encoder is not None:
            raw_enc = text_encoder.module if isinstance(text_encoder, nn.DataParallel) else text_encoder
            payload["text_encoder_state"] = raw_enc.state_dict()

        filename = f"pixeldiffusion_{tag}_epoch_{epoch}.pt"
        torch.save(payload, os.path.join(self.checkpoint_dir, filename))
        
    def _load_weight(self, unet: UNet, optimizer, text_encoder: TextEncoder, pretrained: str):
        full_path = os.path.join(self.checkpoint_dir, pretrained)
        try:
            ckpt = torch.load(full_path, map_location=self.device)

            # Load UNet 
            raw_unet = unet.module if isinstance(unet, nn.DataParallel) else unet
            state_dict = ckpt["unet_state"]

            # Detect if checkpoint is unconditional (no CrossAttention key saved)
            is_uncond_checkpoint = not any("bottleneck.2" in k and "cross" in k.lower() for k in state_dict.keys())

            if is_uncond_checkpoint:
                # Remap
                remapped = {
                    (k.replace("bottleneck.2", "bottleneck.3") if k.startswith("bottleneck.2") else k): v
                    for k, v in state_dict.items()
                }
            else:
                # Cond checkpoint : keys already match
                remapped = state_dict

            raw_unet.load_state_dict(remapped, strict=False)

            # Load optimizer
            if "optimizer_state" in ckpt:
                saved_groups = ckpt["optimizer_state"].get("param_groups", [])
                if len(saved_groups) == len(optimizer.param_groups):
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                else:
                    print(f"Optimizer param group mismatch ({len(saved_groups)} vs {len(optimizer.param_groups)}), skipping optimizer state")

            # Load TextEncoder
            if text_encoder is not None and "text_encoder_state" in ckpt:
                raw_enc = text_encoder.module if isinstance(text_encoder, nn.DataParallel) else text_encoder
                raw_enc.load_state_dict(ckpt["text_encoder_state"], strict=False)

            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.losses = ckpt.get("losses", [])
            print(f"Resuming from epoch {self.start_epoch} ({full_path})")

        except FileNotFoundError:
            print(f"No checkpoint found at {full_path} — training from scratch")