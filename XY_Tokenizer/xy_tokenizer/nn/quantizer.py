import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def sample_vectors(samples, num):
    # samples: (N, D), num_samples: N, feature dim: D
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices].float()  # (num, D), ensure fp32


def kmeans(samples, num_clusters, num_iters=10):
    # samples: (N, D), N samples with D dimensions
    dim, dtype = samples.shape[-1], torch.float32  # Force fp32
    means = sample_vectors(
        samples, num_clusters
    ).float()  # (num_clusters, D), ensure fp32

    for _ in range(num_iters):
        dists = -(
            samples.float().pow(2).sum(1, keepdim=True)  # (N, 1), ensure fp32
            - 2 * samples.float() @ means.t()  # (N, num_clusters), ensure fp32
            + means.t().float().pow(2).sum(0, keepdim=True)
        )  # (1, num_clusters), ensure fp32
        # dists: (N, num_clusters)
        buckets = dists.max(dim=-1).indices  # (N)
        bins = torch.bincount(buckets, minlength=num_clusters)  # (num_clusters)
        zero_mask = bins == 0  # (num_clusters)
        bins_min_clamped = bins.masked_fill(zero_mask, 1)  # (num_clusters)

        new_means = buckets.new_zeros(
            num_clusters, dim, dtype=torch.float32
        )  # (num_clusters, D), ensure fp32
        new_means.scatter_add_(
            0, buckets.unsqueeze(1).expand(-1, dim), samples.float()
        )  # (num_clusters, D), ensure fp32
        new_means = new_means / bins_min_clamped[..., None]  # (num_clusters, D)
        means = torch.where(zero_mask[..., None], means, new_means)  # (num_clusters, D)

    # Final cluster assignments for returning cluster sizes
    dists = -(
        samples.float().pow(2).sum(1, keepdim=True)
        - 2 * samples.float() @ means.t()
        + means.t().float().pow(2).sum(0, keepdim=True)
    )  # (N, num_clusters), ensure fp32
    buckets = dists.max(dim=-1).indices  # (N)
    bins = torch.bincount(
        buckets, minlength=num_clusters
    ).float()  # (num_clusters), ensure fp32

    return means, bins  # (num_clusters, D), (num_clusters)


class VectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        commitment=1.0,
        decay=0.99,  # EMA decay
        epsilon=1e-5,  # Laplace smoothing epsilon
        threshold_ema_dead=2,  # Dead code threshold
        kmeans_init=True,  # Use kmeans initialization
        kmeans_iters=10,  # Kmeans iterations
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead = threshold_ema_dead
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        if self.input_dim != self.codebook_dim:
            self.in_project = WNConv1d(
                self.input_dim, self.codebook_dim, kernel_size=1
            )  # (B, D, T) -> (B, D', T)
            self.out_project = WNConv1d(
                self.codebook_dim, self.input_dim, kernel_size=1
            )  # (B, D', T) -> (B, D, T)
        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        # Initialize codebook and EMA buffers
        init_fn = torch.zeros if kmeans_init else lambda x, y: torch.randn(x, y)
        self.register_buffer(
            "codebook", init_fn(codebook_size, codebook_dim).float()
        )  # (codebook_size, D'), ensure fp32
        self.register_buffer(
            "inited", torch.tensor([not kmeans_init], dtype=torch.bool)
        )  # (1)
        self.register_buffer(
            "cluster_size", torch.zeros(codebook_size).float()
        )  # (codebook_size), ensure fp32
        self.register_buffer(
            "embed_avg", self.codebook.clone().float()
        )  # (codebook_size, D'), ensure fp32

    def ema_update(self, encodings, embed_onehot):
        # encodings: (B*T, D'), embed_onehot: (B*T, codebook_size)
        """Update codebook using EMA"""
        encodings = encodings.float()  # Ensure fp32
        embed_onehot = embed_onehot.float()  # Ensure fp32
        cluster_size_new = embed_onehot.sum(0)  # (codebook_size)
        embed_sum = encodings.t() @ embed_onehot  # (D', codebook_size)

        # Distributed reduction
        if dist.is_initialized():
            dist.all_reduce(cluster_size_new, op=dist.ReduceOp.SUM)
            dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)

        ema_inplace(self.cluster_size, cluster_size_new, self.decay)  # (codebook_size)
        ema_inplace(self.embed_avg, embed_sum.t(), self.decay)  # (codebook_size, D')

        # Laplace smoothing
        cluster_size = (self.cluster_size + self.epsilon) / (
            self.cluster_size.sum() + self.codebook_size * self.epsilon
        )  # (codebook_size)
        cluster_size = cluster_size * self.cluster_size.sum()  # (codebook_size)
        self.codebook.copy_(
            self.embed_avg / cluster_size.unsqueeze(1)
        )  # (codebook_size, D')

    def replace_dead_codes(self, encodings):
        # encodings: (B*T, D')
        """Replace dead codes with random samples from current batch"""
        if self.threshold_ema_dead == 0:
            return

        dead_mask = self.cluster_size < self.threshold_ema_dead  # (codebook_size)
        if dead_mask.any():
            if dist.is_initialized() and dist.get_rank() == 0:
                samples = sample_vectors(
                    encodings.float(), self.codebook_size
                )  # (codebook_size, D'), ensure fp32
            else:
                samples = torch.zeros_like(
                    self.codebook
                ).float()  # Placeholder, ensure fp32

            # Broadcast samples
            if dist.is_initialized():
                dist.broadcast(samples, src=0)

            self.codebook[dead_mask] = samples[: dead_mask.sum()].to(
                self.codebook.dtype
            )  # Update dead codes

    def init_codebook(self, encodings):
        # encodings: (B*T, D')
        """Initialize codebook with k-means and update cluster_size"""
        if self.inited.item():
            return

        if dist.is_initialized() and dist.get_rank() == 0:
            embed, cluster_sizes = kmeans(
                encodings.float(), self.codebook_size, self.kmeans_iters
            )  # (codebook_size, D'), (codebook_size), ensure fp32
        else:
            embed = torch.zeros(
                self.codebook_size, self.codebook_dim, device=encodings.device
            ).float()  # ensure fp32
            cluster_sizes = torch.zeros(
                self.codebook_size, device=encodings.device, dtype=torch.float32
            )  # ensure fp32

        # Broadcast results
        if dist.is_initialized():
            dist.broadcast(embed, src=0)
            dist.broadcast(cluster_sizes, src=0)

        self.codebook.copy_(embed)  # (codebook_size, D')
        self.embed_avg.copy_(embed.clone())  # (codebook_size, D')
        self.cluster_size.copy_(cluster_sizes.float())  # (codebook_size)
        self.inited.fill_(True)

    def forward(self, z):  # z: (B, D, T)
        # logging.info(f"{self.cluster_size = }, {self.codebook = }, {self.embed_avg = }, {self.inited = }")
        z = z.float()  # Ensure fp32
        z_e = self.in_project(z).float()  # (B, D', T), ensure fp32

        # Rearrange for quantization
        encodings = rearrange(z_e, "b d t -> (b t) d").float()  # (B*T, D'), ensure fp32

        # Initialize codebook if needed
        if self.kmeans_init and not self.inited.item():
            self.init_codebook(encodings)

        # Quantization
        dist = (
            encodings.pow(2).sum(1, keepdim=True)  # (B*T, 1)
            - 2 * encodings @ self.codebook.float().t()  # (B*T, codebook_size)
            + self.codebook.float().pow(2).sum(1, keepdim=True).t()
        )  # (1, codebook_size)
        # dist: (B*T, codebook_size)

        indices = (-dist).max(1)[1]  # (B*T)
        indices = rearrange(indices, "(b t) -> b t", b=z.size(0))  # (B, T)

        # Get quantized vectors
        z_q = self.decode_code(indices).float()  # (B, D', T), ensure fp32

        # Commitment loss
        commit_loss = (
            F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
            * self.commitment
        )  # (B)

        # EMA updates and dead code replacement during training
        if self.training and torch.is_grad_enabled():
            embed_onehot = F.one_hot(
                indices.view(-1), self.codebook_size
            ).float()  # (B*T, codebook_size), ensure fp32
            self.ema_update(encodings, embed_onehot)
            self.replace_dead_codes(encodings)

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()  # (B, D', T)
        z_q = self.out_project(z_q).float()  # (B, D, T), ensure fp32

        return (
            z_q,
            commit_loss,
            torch.tensor(0.0, device=z.device, dtype=torch.float32),
            indices,
            z,
        )  # (B, D, T), (B), scalar, (B, T), (B, D', T)

    def decode_code(self, embed_id):  # embed_id: (B, T)
        return (
            F.embedding(embed_id, self.codebook).transpose(1, 2).float()
        )  # (B, D', T), ensure fp32


class ResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,  # Input dimension, unrelated to RVQ
        rvq_dim=None,  # RVQ dimension. If different from input_dim/output_dim, will add input_dim->rvq_dim/rvq_dim->output_dim projection
        output_dim: int = None,  # Output dimension, unrelated to RVQ
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,  # Dimension of each codebook. If different from rvq_dim, will add rvq_dim->codebook_dim and codebook_dim->rvq_dim projections
        quantizer_dropout: float = 0.5,
        decay=0.99,
        epsilon=1e-5,
        threshold_ema_dead=2,
        kmeans_init=True,
        kmeans_iters=10,
        skip_rvq_ratio: float = 0.0,  # New parameter: probability of skipping RVQ
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.skip_rvq_ratio = skip_rvq_ratio  # Store skip probability
        self.rvq_dim = rvq_dim

        self.input_proj = (
            WNConv1d(input_dim, rvq_dim, kernel_size=1)
            if input_dim != rvq_dim
            else nn.Identity()
        )
        self.output_proj = (
            WNConv1d(rvq_dim, output_dim, kernel_size=1)
            if rvq_dim != output_dim
            else nn.Identity()
        )

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(
                    input_dim=rvq_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    decay=decay,
                    epsilon=epsilon,
                    threshold_ema_dead=threshold_ema_dead,
                    kmeans_init=kmeans_init,
                    kmeans_iters=kmeans_iters,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(
        self, z, input_length, n_quantizers: int = None
    ):  # z: (B, D, T), input_length: (B)
        z = self.input_proj(z)

        with torch.autocast("cuda", enabled=False):
            batch_size, _, max_time = z.shape
            mask = torch.arange(max_time, device=z.device).expand(
                batch_size, max_time
            ) < input_length.unsqueeze(
                1
            )  # (B, T)

            quantized_out = torch.zeros_like(
                z, dtype=torch.float32
            )  # (B, D, T), ensure fp32
            residual = z.clone().float()  # (B, D, T), ensure fp32

            all_commit_losses = []
            all_indices = []
            all_quantized = []

            n_quantizers = n_quantizers or self.num_quantizers

            # Randomly decide whether to skip RVQ during training
            skip_mask = None
            if self.training and torch.is_grad_enabled() and self.skip_rvq_ratio > 0:
                # Generate random mask with skip_rvq_ratio probability
                skip_mask = (
                    torch.rand(batch_size, device=z.device) < self.skip_rvq_ratio
                )  # (B,)
                # If all samples are skipped, force the first sample to be unskipped
                if skip_mask.all():
                    skip_mask[0] = (
                        False  # Ensure at least one sample (index 0) is not skipped
                    )

            if self.training and torch.is_grad_enabled():
                n_quantizers_tensor = (
                    torch.ones((z.shape[0],), dtype=torch.float32, device=z.device)
                    * self.num_quantizers
                    + 1
                )  # (B)
                dropout = torch.randint(
                    1,
                    self.num_quantizers + 1,
                    (z.shape[0],),
                    dtype=torch.float32,
                    device=z.device,
                )  # (B)
                n_dropout = int(z.shape[0] * self.quantizer_dropout)
                n_quantizers_tensor[:n_dropout] = dropout[:n_dropout]  # (B)
            else:
                n_quantizers_tensor = torch.full(
                    (z.shape[0],), n_quantizers, dtype=torch.float32, device=z.device
                )  # (B)

            for i, quantizer in enumerate(self.quantizers):
                if not self.training and i >= n_quantizers:
                    break

                masked_residual = residual * mask.unsqueeze(1)  # (B, D, T)

                # If skipping RVQ, directly use input value
                if self.training and skip_mask is not None and skip_mask.any():
                    z_q_i = torch.zeros_like(
                        masked_residual, dtype=torch.float32
                    )  # (B, D, T), ensure fp32
                    commit_loss_i = torch.zeros(
                        batch_size, device=z.device, dtype=torch.float32
                    )  # (B), ensure fp32
                    indices_i = torch.zeros(
                        batch_size, max_time, device=z.device, dtype=torch.long
                    )  # (B, T)
                    z_e_i = torch.zeros_like(
                        masked_residual, dtype=torch.float32
                    )  # (B, D, T), ensure fp32

                    # Quantize non-skipped samples
                    non_skipped_mask = ~skip_mask  # (B)
                    if non_skipped_mask.any():
                        (
                            z_q_i_non_skipped,
                            commit_loss_i_non_skipped,
                            _,
                            indices_i_non_skipped,
                            z_e_i_non_skipped,
                        ) = quantizer(
                            masked_residual[non_skipped_mask].float()  # Ensure fp32
                        )
                        z_q_i[non_skipped_mask] = z_q_i_non_skipped
                        commit_loss_i[non_skipped_mask] = commit_loss_i_non_skipped
                        indices_i[non_skipped_mask] = indices_i_non_skipped
                        z_e_i[non_skipped_mask] = z_e_i_non_skipped
                else:
                    z_q_i, commit_loss_i, _, indices_i, z_e_i = quantizer(
                        masked_residual.float()
                    )  # (B, D, T), (B), scalar, (B, T), (B, D', T), ensure fp32

                quantizer_mask = (
                    torch.full((z.shape[0],), i, device=z.device, dtype=torch.float32)
                    < n_quantizers_tensor
                )  # (B)
                update_mask = (mask & quantizer_mask.unsqueeze(-1)).unsqueeze(
                    1
                )  # (B, 1, T)

                # If skipping, output is directly the input
                if skip_mask is not None:
                    skip_mask_expanded = skip_mask.unsqueeze(1).unsqueeze(
                        2
                    )  # (B, 1, 1)
                    z_q_i = torch.where(
                        skip_mask_expanded, masked_residual, z_q_i
                    )  # (B, D, T)
                    commit_loss_i = torch.where(
                        skip_mask, torch.zeros_like(commit_loss_i), commit_loss_i
                    )  # (B)

                quantized_out = quantized_out + z_q_i * update_mask  # (B, D, T)

                residual_fp32 = residual.to(dtype=torch.float32)  # (B, D, T)
                z_q_i_fp32 = z_q_i.to(dtype=torch.float32)  # (B, D, T)
                residual_fp32 = residual_fp32 - z_q_i_fp32 * update_mask  # (B, D, T)
                residual = residual_fp32.to(
                    dtype=torch.float32
                )  # (B, D, T), ensure fp32

                valid_mask = mask & quantizer_mask.unsqueeze(-1)  # (B, T)
                if valid_mask.any():
                    commit_loss_i = (
                        commit_loss_i * quantizer_mask
                    ).sum() / quantizer_mask.sum()  # scalar
                else:
                    commit_loss_i = torch.tensor(
                        0.0, device=z.device, dtype=torch.float32
                    )  # scalar, ensure fp32

                all_commit_losses.append(commit_loss_i)  # scalar
                all_indices.append(indices_i)  # (B, T)
                all_quantized.append(z_q_i)  # (B, D, T)

            all_commit_losses = torch.stack(all_commit_losses)  # (N)
            all_indices = torch.stack(all_indices)  # (N, B, T)
            all_quantized = torch.stack(all_quantized)  # (N, B, D, T)

            output_length = input_length  # (B)

        quantized_out = self.output_proj(quantized_out)

        return (
            quantized_out,  # (B, D, T)
            all_indices,  # (N, B, T)
            all_commit_losses,  # (N)
            all_quantized,  # (N, B, D, T)
            output_length,  # (B)
        )

    def decode_codes(self, codes):  # codes: (nq, B, T)
        """Decode codes from multiple quantizers to embeddings.

        Args:
            codes: Tensor of shape (nq, B, T) containing code indices for each quantizer.

        Returns:
            emb: Tensor of shape (B, D, T) representing the decoded embeddings.
        """
        nq, B, T = codes.shape
        device = codes.device
        emb = torch.zeros(
            B, self.rvq_dim, T, device=device, dtype=torch.float32
        )  # (B, D, T)

        for i, quantizer in enumerate(self.quantizers[:nq]):
            code_i = codes[i]  # (B, T)
            quantized_i = quantizer.decode_code(code_i)  # (B, D', T)
            emb += quantized_i  # Accumulate quantized embeddings

        emb = self.output_proj(emb)  # (B, D, T), apply output projection
        return emb  # (B, D, T)


def ema_inplace(moving_avg, new, decay):
    # moving_avg: (codebook_size) or (codebook_size, D'), new: same as moving_avg
    """Update exponential moving average in-place"""
    moving_avg.data.mul_(decay).add_(new.float(), alpha=(1 - decay))  # ensure fp32
