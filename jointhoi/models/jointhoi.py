import math

import torch
from torch import nn


class JointHOI(nn.Module):
    """Transformer denoiser over concatenated motion and dual distance maps.

    The module names intentionally match the research checkpoints.
    """

    def __init__(
        self,
        hand_nfeats=99,
        obj_nfeats=10,
        dmap_nfeats=2048,
        cond_dim=1092,
        latent_dim=2048,
        ff_size=4096,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        max_nframes=64,
        num_train_timesteps=1000,
        **kwargs,
    ):
        super().__init__()
        self.hand_nfeats = hand_nfeats
        self.obj_nfeats = obj_nfeats
        self.dmap_nfeats = dmap_nfeats
        self.latent_dim = latent_dim
        self.total_input_feats = 2 * hand_nfeats + obj_nfeats + dmap_nfeats

        self.null_dmap = nn.Parameter(torch.zeros(1, 1, dmap_nfeats))
        nn.init.normal_(self.null_dmap, mean=0.0, std=1e-6)

        max_length = max(max_nframes + 2, num_train_timesteps)
        self.input_process = nn.Linear(self.total_input_feats, latent_dim)
        self.output_lhand = nn.Linear(latent_dim, hand_nfeats)
        self.output_rhand = nn.Linear(latent_dim, hand_nfeats)
        self.output_obj = nn.Linear(latent_dim, obj_nfeats)
        self.output_dmap = nn.Linear(latent_dim, dmap_nfeats)

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout, max_length)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        self.condition_encoder = nn.Sequential(
            nn.Linear(cond_dim + 512, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        x_lhand,
        x_rhand,
        x_obj,
        x_dmap,
        obj_feat_final,
        timestep,
        enc_text,
        valid_mask_lhand=None,
        valid_mask_rhand=None,
        valid_mask_obj=None,
    ):
        batch_size, frames = x_lhand.shape[:2]
        joint_input = torch.cat([x_lhand, x_rhand, x_obj, x_dmap], dim=-1)
        frame_tokens = self.input_process(joint_input)
        time_token = self.embed_timestep(timestep)
        condition = torch.cat([obj_feat_final, enc_text], dim=1)
        condition_token = self.condition_encoder(condition).unsqueeze(1)
        sequence = torch.cat([condition_token, time_token, frame_tokens], dim=1)
        sequence = self.sequence_pos_encoder(sequence)

        padding_mask = None
        masks = (valid_mask_lhand, valid_mask_rhand, valid_mask_obj)
        if any(mask is not None for mask in masks):
            valid = torch.zeros(batch_size, frames, dtype=torch.bool, device=x_lhand.device)
            for mask in masks:
                if mask is not None:
                    valid |= mask
            prefix_valid = torch.ones(batch_size, 2, dtype=torch.bool, device=x_lhand.device)
            padding_mask = ~torch.cat([prefix_valid, valid], dim=1)

        output = self.seqTransEncoder(sequence, src_key_padding_mask=padding_mask)[:, 2:]
        return (
            self.output_lhand(output),
            self.output_rhand(output),
            self.output_obj(output),
            self.output_dmap(output),
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(1, max_len, d_model)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", encoding)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[0, timesteps]).unsqueeze(1)


# Name retained for scripts/checkpoints from the research workspace.
TextHOM_Joint_DDIM_Fix = JointHOI
