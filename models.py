from collections import defaultdict
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
from tqdm import tqdm
import numpy as np
import mlx.core as mx

from cube_model import RubiksCube
from data_utils import RunningMean, RunningMin


# ---------------------------------------------------------
# Dataset Generator
# ---------------------------------------------------------


class RubiksCubeData(Dataset):
    '''A class to represent the Rubik's cube dataset. This class performs random walks in the
    Rubik's cube group beginning at the solved state to generate pairs `(s, l)`, where `s` is
    a Rubik's cube state and `l` is the mean (or minimum) diffusion distance from the solved
    state to `s`.
    '''

    def __init__(
        self,
        k: int,
        n: Optional[int] = None,
        total: Optional[int] = None,
        method: str = 'mean',
        dtype: torch.dtype | np.dtype | mx.Dtype = torch.float32,
        target_dtype: torch.dtype | np.dtype | mx.Dtype = torch.float32,
        scale: bool = True,
        one_hot: bool = True,
        verbose: bool = True,
    ):
        '''
        Parameters
        ----------
        k : int
            The number of steps to take in each random walk.
        n : Optional[int]
            The number of random walks to perform. If None, `total` must be specified.
        total : Optional[int]
            The total number of training samples to generate. If None, `n` must be specified.
        method : str
            The method for computing distance from a particular configuration to the solved state.
            Must be either 'mean' or 'min'.
        dtype : torch.dtype | np.dtype | mx.Dtype
            The data type for the tensors.
        target_dtype : torch.dtype | np.dtype | mx.Dtype
            The data type for the targets.
        scale : bool
            If True, scale the targets into [0, 1].
        one_hot : bool
            Whether or not to one-hot encode the data.
        verbose : bool
            Whether or not do display progress bars.
        '''

        super().__init__()

        if n is None:
            if total is None:
                raise ValueError('One of `n` or `total` must be supplied.')
            n = total
        self.k = k
        self.n = n
        self.total = total
        self.method = method
        self.dtype = dtype
        self.target_dtype = target_dtype
        self.scale = scale
        self.one_hot = one_hot
        self.verbose = verbose

        self.x: Optional[torch.Tensor | np.ndarray | mx.array] = None
        self.y: Optional[torch.Tensor | np.ndarray | mx.array] = None

    def generate(self) -> 'RubiksCubeData':
        '''Perform random walks beginning from the solved state to generate pairs `(s, l)`, where `s`
        is the cube state and `l` is the diffusion distance from `s` to the solved state.

        Returns
        -------
        self : RubiksCubeData
        '''

        accumulator = {'mean': RunningMean, 'min': RunningMin}[self.method]

        # Map the cube states (given as Python ints) to the computed distance to the solved state
        distance_to_solved = defaultdict(lambda: accumulator())

        def random_walk():
            cube = RubiksCube()
            previous_state = cube.state
            # Perform the random walk
            for step in range(1, self.k + 1):
                # Store the state of the current configuration
                new_previous_state = cube.state
                # Perform a random move on the cube
                move = cube.get_random_move()
                cube._move(move)
                # If the random move inverts the previous one, undo it and try another
                while cube.state == previous_state:
                    cube._move(cube.inverse(move))
                    move = cube.get_random_move()
                    cube._move(move)
                # Store the previous state of the cube as the new previous state
                previous_state = new_previous_state
                # Update the accumulator for the new cube
                distance_to_solved[cube.state].update(step)

        if self.total is None:
            for _ in tqdm(
                range(self.n), desc=f'Generating train data for k={self.k}, n={self.n:_}', disable=not self.verbose
            ):
                random_walk()
        else:
            with tqdm(
                desc=f'Generating train data for k={self.k}, total={self.total:_}',
                disable=not self.verbose,
                total=self.total,
            ) as pbar:
                length = 0
                while True:
                    random_walk()
                    new_len = len(distance_to_solved)
                    pbar.update(new_len - length)
                    length = new_len
                    if length >= self.total:
                        break

        if self.verbose:
            print('Converting data to arrays...')
        if isinstance(self.dtype, torch.dtype):
            self.y = torch.tensor([acc.get() for acc in distance_to_solved.values()], dtype=self.target_dtype)
            self.x = RubiksCube._to_tensor(distance_to_solved.keys(), self.dtype, self.one_hot)
        elif isinstance(self.dtype, mx.Dtype):
            self.y = mx.array([acc.get() for acc in distance_to_solved.values()], dtype=self.target_dtype)
            self.x = RubiksCube._to_mxarray(distance_to_solved.keys(), self.dtype, self.one_hot)
        else:
            self.y = np.fromiter((acc.get() for acc in distance_to_solved.values()), dtype=self.target_dtype)
            self.x = RubiksCube._to_ndarray(distance_to_solved.keys(), self.dtype, self.one_hot)

        if self.total is not None:
            self.x = self.x[: self.total]
            self.y = self.y[: self.total]

        if self.scale:
            self.y /= self.k

        return self

    def shuffle(self) -> 'RubiksCubeData':
        '''Shuffle the dataset.

        Returns
        -------
        self : RubiksCubeData
        '''

        if self.x is not None:
            idx = torch.randperm(self.x.shape[0])
            if isinstance(self.dtype, mx.Dtype):
                idx = mx.array(idx.numpy())
            self.x = self.x[idx]
            self.y = self.y[idx]
        return self

    def offload(self):
        '''Offload the stored data from memory.'''

        self.x = None
        self.y = None

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class IterableRubiksCubeData(IterableDataset):
    '''Iteration support for the Rubiks cube dataset without storing all data in memory.
    Data is generated on-the-fly in batches of `batch_generation_size`.
    '''

    def __init__(
        self,
        k: int,
        total: int,
        batch_generation_size: int,
        method: str = 'mean',
        dtype: torch.dtype | np.dtype | mx.Dtype = torch.float32,
        target_dtype: torch.dtype | np.dtype | mx.Dtype = torch.float32,
        scale: bool = True,
        one_hot: bool = True,
    ):
        '''
        k : int
            The number of steps to take in each random walk.
        total : int
            The total number of training samples to generate.
        batch_generation_size : int
            The maximum number of training samples to store in memory at any one time.
        method : str
            The method for computing distance from a particular configuration to the solved state.
            Must be either 'mean' or 'min'.
        dtype : torch.dtype | np.dtype | mx.Dtype
            The data type for the tensors.
        target_dtype : torch.dtype | np.dtype | mx.Dtype
            The data type for the targets.
        scale : bool
            If True, scale the targets into [0, 1].
        one_hot : bool
            Whether or not to one-hot encode the data.
        '''
        super().__init__()

        self.total = total
        self.dataset = RubiksCubeData(
            k=k,
            n=None,
            total=batch_generation_size,
            method=method,
            dtype=dtype,
            target_dtype=target_dtype,
            scale=scale,
            one_hot=one_hot,
            verbose=False,
        )

    def __iter__(self):
        idx = 0
        self.dataset.generate()
        for _ in range(self.total):
            if idx >= self.dataset.total:
                self.dataset.generate()
                idx = 0
            yield self.dataset[idx]
            idx += 1

    def offload(self):
        '''Offload the stored data from memory.'''

        self.dataset.offload()

    def __len__(self):
        return self.total


class PairwiseRubiksCubeDataset(Dataset):
    '''Generates pairs `(closet, farther)` where `closer` and `farther` are cube states with the
    diffusion distance of `closer` to the solved state being less than that of `farther`.
    '''

    def __init__(
        self,
        k: int,
        total: int,  # Total number of pairs
        batch_generation_size: int,
        method: str = 'mean',
        dtype: torch.dtype | np.dtype | mx.Dtype = torch.float32,
        target_dtype: torch.dtype | np.dtype | mx.Dtype = torch.float32,
        scale: bool = True,
        one_hot: bool = True,
        shuffle: bool = True,
        verbose: bool = True,
    ):
        '''
        k : int
            The number of steps to take in each random walk.
        total : int
            The total number of data samples to generate.
        batch_generation_size : int
            The maximum number of data samples to store in memory at any one time.
        method : str
            The method for computing distance from a particular configuration to the solved state.
            Must be either 'mean' or 'min'.
        dtype : torch.dtype | np.dtype | mx.Dtype
            The data type for the tensors.
        target_dtype : torch.dtype | np.dtype | mx.Dtype
            The data type for the targets.
        scale : bool
            If True, scale the targets into [0, 1].
        one_hot : bool
            Whether or not to one-hot encode the data.
        shuffle : bool
            Whether or not to shuffle the dataset.
        verbose : bool
            Whether or not do display progress bars.
        '''
        super().__init__()

        self.total = total
        self.shuffle = shuffle
        self.dataset = RubiksCubeData(
            k=k,
            n=None,
            total=batch_generation_size,
            method=method,
            dtype=dtype,
            target_dtype=target_dtype,
            scale=scale,
            one_hot=one_hot,
            verbose=verbose,
        )
        self._idx = 0

    def _load_new_dataset(self):
        self.dataset.generate()
        if self.shuffle:
            self.dataset.shuffle()

    def _get_next(self):
        if self.dataset.x is None:
            self._load_new_dataset()
        try:
            x, y = self.dataset.x[self._idx], self.dataset.y[self._idx]
        except IndexError:
            self._load_new_dataset()
            self._idx = 0
            x, y = self.dataset.x[self._idx], self.dataset.y[self._idx]
        self._idx += 1
        return x, y

    def __getitem__(self, _: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        a, b = self._get_next(), self._get_next()
        if a[1] < b[1]:
            return a, b
        return b, a

    def offload(self):
        '''Offload the stored data from memory.'''

        self.dataset.offload()

    def __len__(self):
        return self.total


# ---------------------------------------------------------
# MLP with Residual Connections
# ---------------------------------------------------------


class ResBlock(nn.Module):
    '''A residual block.'''

    def __init__(self, dim: int, dropout_rate: float, activation_func: nn.Module, use_batch_norm: bool):
        '''
        Parameters
        ----------
        dim : int
            The dimension of the linear layers.
        dropout_rate : float
        activation_func : nn.Module
        use_batch_norm : bool
            If True, a batch normalization layer will be placed after each linear layer.
        '''

        super().__init__()

        self.activation_func = activation_func()
        layers = [
            nn.Linear(dim, dim),
            self.activation_func,
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        ]
        if use_batch_norm:
            layers.insert(1, nn.BatchNorm1d(dim))
            layers.append(nn.BatchNorm1d(dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_func(self.layers(x) + x)


class MLP(nn.Module):
    '''A multi-layer perceptron.'''

    def __init__(
        self,
        in_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        num_res_blocks: int,
        dropout_rate: float = 0.1,
        activation_func: nn.Module = nn.ReLU,
        use_batch_norm: bool = True,
    ):
        '''
        Parameters
        ----------
        in_dim : int
            The dimension of the input data.
        hidden_dim_1, hidden_dim_2 : int
            The dimensions of the hidden layers.
        num_res_blocks : int
            The number of residual blocks.
        dropout_rate : float
        activation_func : nn.Module
        use_batch_norm : bool
            If True, a batch normalization layer will be placed after each linear layer.
        '''
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(in_dim, hidden_dim_1))
        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_dim_1))
        self.layers.extend([activation_func(), nn.Dropout(dropout_rate), nn.Linear(hidden_dim_1, hidden_dim_2)])
        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_dim_2))
        self.layers.extend([activation_func(), nn.Dropout(dropout_rate)])
        self.layers.extend(
            ResBlock(hidden_dim_2, dropout_rate, activation_func, use_batch_norm) for _ in range(num_res_blocks)
        )
        self.layers.extend([nn.Linear(hidden_dim_2, 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layers(x))


# ---------------------------------------------------------
# Embedding MLP
# ---------------------------------------------------------


class EmbeddingMLP(nn.Module):
    '''A multi-layer perceptron which embeds each input feature as a vector of size `embedding_dim`.'''

    def __init__(
        self,
        n_features: int,
        n_categories: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout_rate: float,
        activation_func: nn.Module,
    ):
        '''
        Parameters
        ----------
        n_features : int
            The number of dataset categorical features.
        n_categories : int
            The number of categories for each feature.
        embedding_dim : int
            The dimension of the embedding vectors.
        hidden_dim : int
            The dimension of the hidden layer.
        n_layers : int
            The number o fhidden layers.
        dropout_rate : float
        acitvation_func : nn.Module
        '''

        super().__init__()

        self.n_features = n_features
        self.n_categories = n_categories

        self.embeddings = nn.Embedding(n_features * n_categories, embedding_dim)
        self.register_buffer('offsets', torch.arange(n_features) * n_categories)

        in_dim = embedding_dim * n_features

        self.mlp = nn.Sequential()
        for i in range(n_layers):
            self.mlp.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            self.mlp.append(activation_func())
            self.mlp.append(nn.Dropout(dropout_rate))
        self.mlp.append(nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x + self.offsets)  # shape: (batch, n_features, embedding_dim)
        x = x.view(x.size(0), -1)  # flatten to (batch, n_features * embedding_dim)
        return torch.sigmoid(self.mlp(x))


# ---------------------------------------------------------
# Transformer Model for Tabular Data
# ---------------------------------------------------------


class FeatureTokenizer(nn.Module):
    '''Tokenizes categorical features into dense embeddings. Each feature is represented by a
    learnable embedding vector, and an additional feature-specific embedding is added to
    distinguish which feature is which.
    '''

    def __init__(self, n_features: int, n_classes: int, d_model: int):
        '''
        Parameters
        ----------
        n_features : int
            The number of features in the dataset.
        n_classes : int
            The number of categories for each feature.
        d_model : int
            The dimension of the embedding vectors.
        '''

        super().__init__()

        self.embeddings = nn.Embedding(n_classes, d_model)
        # Feature ID embeddings help model know “which feature is which”
        self.feature_emb = nn.Parameter(torch.randn(n_features, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x) + self.feature_emb[None, :, :]  # (batch, n_features, d_model)


class TransformerBlock(nn.Module):
    '''A standard Transformer block consisting of multi-head self-attention, a feed-forward
    network, and residual connections with layer normalization.
    '''

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float):
        '''
        Parameters
        ----------
        d_model : int
            The dimensionality of the input and output embeddings.
        n_heads : int
            The number of attention heads in the multi-head self-attention mechanism.
        d_ff : int
            The dimensionality of the hidden layer in the feed-forward network.
        dropout_rate : float
            The dropout probability applied after attention and feed-forward layers.
        '''

        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout_rate, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.layer_norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.layer_norm2(x + ff_out)
        return x


class FTTransformer(nn.Module):
    '''A Transformer-based model for tabular data. Input features are embedded into tokens,
    passed through multiple Transformer blocks, pooled, and projected to a single output.
    '''

    def __init__(
        self, n_features: int, n_classes: int, d_model: int, n_heads: int, d_ff: int, depth: int, dropout_rate: float
    ):
        '''
        Parameters
        ----------
        n_features : int
            The number of features in the dataset.
        n_classes : int
            The number of categories for each feature.
        d_model : int
            The dimension of the embedding vectors and model hidden states.
        n_heads : int
            The number of attention heads in each Transformer block.
        d_ff : int
            The hidden dimension of the feed-forward layers in each Transformer block.
        depth : int
            The number of stacked Transformer blocks.
        dropout_rate : float
            The dropout probability applied in attention and feed-forward layers.
        '''

        super().__init__()

        self.tokenizer = FeatureTokenizer(n_features, n_classes, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout_rate) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        pooled = tokens.mean(dim=1)
        out = self.head(self.norm(pooled))
        return torch.sigmoid(out)


# ---------------------------------------------------------
# Rank Model
# ---------------------------------------------------------


class SwiGLUBlock(nn.Module):
    '''A feed-forward block using the SwiGLU activation function. Consists of a layer
    normalization, a gated linear unit, and a residual connection.
    '''

    def __init__(self, dim: int, hidden_dim: int):
        '''
        Parameters
        ----------
        dim : int
            The dimensionality of the input and output vectors.
        hidden_dim : int
            The hidden dimension of the SwiGLU feed-forward layer (before gating).
        '''

        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h, g = self.fc1(self.norm(x)).chunk(2, dim=-1)
        x = self.fc2(F.silu(h) * g)
        return residual + x


class MLPEncoder(nn.Module):
    '''An encoder that embeds categorical features and processes them with a stack of
    SwiGLU blocks followed by a projection to a fixed embedding dimension.
    '''

    def __init__(
        self, n_features: int, n_categories: int, cat_embed_dim: int, hidden_dim: int, embed_dim: int, n_layers: int
    ):
        '''
        Parameters
        ----------
        n_features : int
            The number of categorical features in the dataset.
        n_categories : int
            The number of categories for each feature.
        cat_embed_dim : int
            The dimension of the categorical embedding vectors.
        hidden_dim : int
            The hidden dimension of each SwiGLU block.
        embed_dim : int
            The output embedding dimension of the encoder.
        n_layers : int
            The number of SwiGLU blocks in the encoder.
        '''

        super().__init__()

        self.categorical_embedding = nn.Embedding(n_features * n_categories, cat_embed_dim)
        self.register_buffer('offsets', torch.arange(n_features) * n_categories)

        self.mlp = nn.Sequential(SwiGLUBlock(n_features * cat_embed_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.mlp.append(SwiGLUBlock(n_features * cat_embed_dim, hidden_dim))
        self.mlp.append(nn.Linear(n_features * cat_embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.categorical_embedding(x + self.offsets)
        return self.mlp(x.view(x.shape[0], -1))


class RankModel(nn.Module):
    '''A ranking model that embeds Rubik's cube states and computes their similarity
    to a fixed anchor state. The model is trained with a margin ranking loss so that
    states closer to the solved state have smaller distances to the anchor.
    '''

    def __init__(self, encoder: nn.Module, anchor_state: torch.Tensor):
        '''
        Parameters
        ----------
        encoder : nn.Module
            The encoder network used to compute embeddings of cube states.
        anchor_state : torch.Tensor
            The Rubik's cube state representing the solved configuration, used as the anchor.
        '''

        super().__init__()

        self.encoder = encoder
        with torch.no_grad():
            self.register_buffer('anchor_embedding', self.encoder(anchor_state.view(1, -1)))

    def distance(self, state: torch.Tensor) -> torch.Tensor:
        '''Compute the cosine distance in embedding space between the input state and the solved state.'''

        state_embedding = self.encoder(state)
        sim = F.cosine_similarity(state_embedding, self.anchor_embedding.expand_as(state_embedding))
        return -sim

    def forward(self, closer_state: torch.Tensor, farther_state: torch.Tensor) -> torch.Tensor:
        d_closer = self.distance(closer_state)
        d_farther = self.distance(farther_state)
        loss = F.relu(1.0 + d_closer - d_farther).mean()
        return loss


def batch_infer(model: MLP | EmbeddingMLP, x: torch.Tensor, batch_size: int):
    '''Infer the MLP model in batches.

    Parameters
    ----------
    model : MLP
        An MLP model to perform inference on.
    x : torch.Tensor
        An input tensor.
    batch_size : int
        The batch size to use.

    Returns
    -------
    out : torch.Tensor
        The result of applying the model to the input tensor.
    '''

    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    model.eval()

    outputs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            outputs.append(model(batch).squeeze(-1).cpu())
    return torch.cat(outputs, dim=0)
