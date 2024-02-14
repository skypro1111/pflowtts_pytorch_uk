from abc import ABC

import torch
import torch.nn.functional as F

from pflow.models.components.decoder import Decoder
from pflow.models.components.wn_pflow_decoder import DiffSingerNet
from pflow.models.components.vits_wn_decoder import VitsWNDecoder

from pflow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, cond=None, training=False, guidance_scale=0.0):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, cond=cond, training=training, guidance_scale=guidance_scale)

    def solve_euler(self, x, t_span, mu, mask,  cond, training=False, guidance_scale=0.0):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        steps = 1
        while steps <= len(t_span) - 1:
            dphi_dt = self.estimator(x, mask, mu, t, cond, training=training)
            if guidance_scale > 0.0:
                mu_avg = mu.mean(2, keepdims=True).expand_as(mu)
                dphi_avg = self.estimator(x, mask, mu_avg, t, cond, training=training)
                dphi_dt = dphi_dt + guidance_scale * (dphi_dt - dphi_avg)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]
        
    def compute_loss(self, x1, mask, mu, cond=None, training=True, loss_mask=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)
                
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        # y = u * t + z
        estimator_out = self.estimator(y, mask, mu, t.squeeze(), training=training)

        if loss_mask is not None:
            mask = loss_mask
        loss = F.mse_loss(estimator_out*mask, u*mask, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
        )

        # Just change the architecture of the estimator here
        self.estimator = Decoder(in_channels=in_channels*2, out_channels=out_channel, **decoder_params)
        # self.estimator = DiffSingerNet(in_dims=in_channels, encoder_hidden=out_channel)
        # self.estimator = VitsWNDecoder(
        #     in_channels=in_channels,
        #     out_channels=out_channel,
        #     hidden_channels=out_channel,
        #     kernel_size=3,
        #     dilation_rate=1,
        #     n_layers=18,
        #     gin_channels=out_channel*2
        # )
        
