import torch
from models.unet import UNet
from models.text_encoder import TextEncoder

class Loader:
    def __init__(self, checkpoint_path: str, device=None, n_heads: int = 4):
        """
        checkpoint_path : path to the .pt checkpoint file
        device : torch.device (auto-detected if None)
        n_heads : number of attention heads (not stored in checkpoint, must match training)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_heads = n_heads

    def load(self) -> tuple:
        """
        Loads a checkpoint and returns (unet, text_encoder, meta).

        unet         : UNet ready for inference (eval mode)
        text_encoder : TextEncoder ready for inference, or None if unconditional checkpoint
        meta         : dict with epoch, loss, losses, conditioned
        """
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        unet_state = ckpt["unet_state"]

        # Detect conditioned vs unconditioned from checkpoint keys
        # CrossAttention sits at bottleneck[2] and owns q_proj/k_proj/v_proj
        is_conditioned = any("bottleneck.2.q_proj" in k for k in unet_state.keys())

        # Infer UNet hyper-params directly from weight shapes
        in_channels = unet_state["init_conv.weight"].shape[1]
        base_channels = unet_state["init_conv.weight"].shape[0]
        emb_dim = unet_state["encoder_blocks.0.0.time_proj.1.weight"].shape[1]
        channel_mults = self._infer_channel_mults(unet_state, base_channels)
        context_dim = (
            unet_state["bottleneck.2.k_proj.weight"].shape[1]
            if is_conditioned else None
        )

        # Build and load UNet
        unet = UNet(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            n_heads=self.n_heads,
            emb_dim=emb_dim,
            context_dim=context_dim,
        ).to(self.device)

        unet.load_state_dict(unet_state, strict=True)
        unet.eval()

        # Build and load TextEncoder (conditioned checkpoints only)
        text_encoder = None
        if is_conditioned and "text_encoder_state" in ckpt:
            enc_state = ckpt["text_encoder_state"]
            vocab_size = enc_state["token_emb.weight"].shape[0]
            dim = enc_state["token_emb.weight"].shape[1]
            max_len = enc_state["pos_emb.weight"].shape[0]
            n_layers = sum(
                1 for k in enc_state
                if k.startswith("blocks.") and k.endswith(".norm1.weight")
            )

            text_encoder = TextEncoder(
                vocab_size=vocab_size,
                dim=dim,
                n_layers=n_layers,
                n_heads=self.n_heads,
                max_len=max_len,
            ).to(self.device)
            text_encoder.load_state_dict(enc_state, strict=True)
            text_encoder.eval()

        meta = {
            "epoch" : ckpt.get("epoch", 0),
            "loss" : ckpt.get("loss", None),
            "losses" : ckpt.get("losses", []),
            "conditioned": is_conditioned,
        }

        print(f"Loaded {'conditioned' if is_conditioned else 'unconditional'} checkpoint "
              f"(epoch {meta['epoch']}) from {self.checkpoint_path}")

        return unet, text_encoder, meta

    def _infer_channel_mults(self, unet_state: dict, base_channels: int) -> tuple:
        """Reconstruct channel_mults by reading encoder block output shapes."""
        mults, i = [], 0
        while f"encoder_blocks.{i}.0.block1.2.weight" in unet_state:
            out_ch = unet_state[f"encoder_blocks.{i}.0.block1.2.weight"].shape[0]
            mults.append(out_ch // base_channels)
            i += 1
        return tuple(mults)
