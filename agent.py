from typing import Optional, Any
from collections.abc import Callable, MutableMapping
from pathlib import Path
from functools import partial

import dill
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim

from cube_model import RubiksCube
from models import (
    MLPEncoder,
    PairwiseRubiksCubeDataset,
    RankModel,
    RubiksCubeData,
    batch_infer,
    MLP,
    EmbeddingMLP,
    FTTransformer,
)
from models_mlx import MLXEmbeddingMLP, batch_infer_mlx
from beam_search import beam_search


class Agent:
    '''A base class for Rubik's cube agents. Each agent is responsible for training a model relating
    Rubik's cube states to diffusion distances, and running the beam search algorithm to search for
    solutions to scrambled cubes. This is equivalent to path finding in large Cayley graphs.
    '''

    def __init__(
        self,
        beam_size: int,
        max_search_its: int,
        rw_length: int,
        train_size: int,
        rw_accumulation_method: str,
        scale_data: bool,
        one_hot: bool,
        dtype: torch.dtype | np.dtype | mx.Dtype,
        target_dtype: torch.dtype | np.dtype | mx.Dtype,
        verbose: bool,
    ):
        '''
        Parameters
        ----------
        beam_size : int
            The size of the beam used in the beasm search.
        max_search_its : int
            The maximum number of iterations for the beam search algorith.
        rw_length : int
            The length of the random walk to use when generating Rubik's cube data samples.
        train_size : int
            The size of the training set to generate.
        rw_accumulation_method : str
            The method for computing distance from a particular Rubik's cube state to the solved state.
            Must be either 'mean' or 'min'.
        scale_data : bool
            If True, scale the data targets into [0, 1].
        one_hot : bool
            Whether or not to one-hot encode the data.
        dtype : torch.dtype | np.dtype | mx.Dtype
            The data type for the tensors.
        target_dtype : torch.dtype | np.dtype | mx.Dtype
            The data type for the targets.
        verbose : bool
            Whether or not do display progress bars.
        '''

        self.beam_size = beam_size
        self.max_search_its = max_search_its
        self.rw_length = rw_length
        self.train_size = train_size
        self.rw_accumulation_method = rw_accumulation_method
        self.scale_data = scale_data
        self.one_hot = one_hot
        self.dtype = dtype
        self.target_dtype = target_dtype
        self.verbose = verbose

        self.data = RubiksCubeData(
            k=self.rw_length,
            total=self.train_size,
            method=self.rw_accumulation_method,
            dtype=self.dtype,
            target_dtype=target_dtype,
            scale=self.scale_data,
            one_hot=self.one_hot,
            verbose=self.verbose,
        )

    def save(self, offload_data: bool = True, prefix: str = ''):
        '''Save the agent to a pickle file.

        Parameters
        ----------
        offload_data : bool
            If True, offload the dataset before saving.
        prefix : str
            A prefix for the agent file name.
        '''

        if offload_data:
            self.data.offload()

        folder = Path.cwd() / 'agent_cache'
        folder.mkdir(exist_ok=True)

        idx = id(self)
        file_name = f'{prefix}_{idx}.pkl' if prefix else f'{idx}.pkl'
        while (folder / file_name).exists():
            idx += 1
            file_name = f'{prefix}_{idx}.pkl' if prefix else f'{idx}.pkl'

        with (folder / file_name).open('wb') as f:
            dill.dump(self, f)

    @classmethod
    def from_path(cls, path: Path | str) -> 'Agent':
        '''Load an agent from a path.

        Parameters
        ----------
        path : Path | str
            The path to load the agent from.
        '''

        with open(path, 'rb') as f:
            agent = dill.load(f)
        return agent

    def _search(self, start_cube: RubiksCube, score_fn: Callable[[list[RubiksCube]], list[float]]) -> list[str] | None:
        '''Perform a beam search to solve the cube.

        Parameters
        ----------
        start_state : RubiksCube
            The scrambled state of the rubiks cube to solve.
        score_fn : Callable[[list[RubiksCube]], list[float]]
            A function which takes a list of cubes and returns a list of scores denoting the predicted distance
            of the cubes to the solved states.

        Returns
        -------
        list[str] | None
            The sequence of moves that solves the cube, or None is no such sequence is found.
        '''

        goal_fn = lambda cube: cube.is_solved()
        successor_fn = lambda cube: cube.iter_neighbours()

        return RubiksCube.get_move_sequence(
            beam_search(
                start_state=start_cube.copy(),
                successor_fn=successor_fn,
                score_fn=score_fn,
                goal_fn=goal_fn,
                beam_size=self.beam_size,
                max_its=self.max_search_its,
                is_batch_score_fn=True,
                verbose=self.verbose,
            )
        )

        # return RubiksCube.get_move_sequence(
        #     a_star_vectorized(
        #         start_state=start_cube,
        #         successor_fn=self.successor_fn,
        #         score_fn=score_fn,
        #         goal_fn=goal_fn,
        #         batch_size=1000,
        #     )
        # )

    def search(self, start_cube: RubiksCube) -> list[str] | None:
        raise NotImplementedError

    @staticmethod
    def score(targets: np.ndarray, predictions: np.ndarray, plot: bool = True):
        '''Score the model and generate plots based on targets and predictions.'''

        r2 = r2_score(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = root_mean_squared_error(targets, predictions)

        print(f'R2: {r2:.4f}  |  MSE: {mse:.4f}  |  RMSE: {rmse:.4f}')

        if plot:
            _, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 8))
            ax0.scatter(targets, predictions, s=2)
            xlim = ax0.get_xlim()
            ylim = ax0.get_ylim()
            lims = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
            ax0.plot(lims, lims, c='r')
            ax0.set_xlim(*lims)
            ax0.set_ylim(*lims)
            ax0.set_xlabel('Targets')
            ax0.set_ylabel('Predictions')

            residuals = targets - predictions
            ax1.scatter(targets, residuals, s=2)
            ylim = ax1.get_ylim()
            ax1.plot(lims, [0, 0], c='r')
            ax1.set_xlim(*lims)
            ax1.set_xlabel('Targets')
            ax1.set_ylabel('Residuals')

            plt.tight_layout()
            plt.show()


class MLPAgent(Agent):
    '''A Rubik's cube agent using an MLP. This is the model described in https://arxiv.org/abs/2502.13266'''

    def __init__(
        self,
        beam_size: int,
        max_search_its: int,
        rw_length: int,
        train_size: int,
        rw_accumulation_method: str,
        scale_data: bool,
        one_hot: bool,
        dtype: torch.dtype,
        target_dtype: torch.dtype,
        verbose: bool,
        # MLPAgent specific params:
        hidden_dim_1: int,
        hidden_dim_2: int,
        num_res_blocks: int,
        activation_func: nn.Module,
        dropout_rate: float,
        use_batch_norm: bool,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optimizer: optim.Optimizer,
        optimizer_params: dict[str, Any],
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        lr_scheduler_params: Optional[dict[str, Any]],
        loss_func: nn.modules.loss._Loss,
        device: torch.device,
    ):
        '''
        Parameters
        ----------
        beam_size, max_search_its, rw_length, train_size, rw_accumulation_method,
        scale_data, one_hot, dtype, target_dtype, verbose
            See `Agent` class.
        hidden_dim_1, hidden_dim_2, num_res_blocks, activation_func, dropout_rate, use_batch_norm
            See MLP class.
        epochs : int
            The number of epochs to train for.
        batch_size : int
            The batch size for model training.
        learning_rate : float
            The learning rate for model training.
        optimizer : optim.Optimizer
            The optimizer to optimize model parameters.
        optimizer_params : dict[str, Any]
            A dictionary of parameters for the optimizer, not including learning rate.
        lr_scheduler : Optional[optim.lr_scheduler.LRScheduler]
            Optionally, an LR scheduler.
        lr_scheduler_params : Optional[dict[str, Any]]
            A dictionary of parameters for the LR scheduler.
        loss_func : nn.modules.loss._Loss
            The loss function to train the model.
        device : torch.device
            The device to train the model on.
        '''

        super().__init__(
            beam_size=beam_size,
            max_search_its=max_search_its,
            rw_length=rw_length,
            train_size=train_size,
            rw_accumulation_method=rw_accumulation_method,
            scale_data=scale_data,
            one_hot=one_hot,
            dtype=dtype,
            target_dtype=target_dtype,
            verbose=verbose,
        )

        self.model = MLP(
            6 * 6 * 8,
            hidden_dim_1,
            hidden_dim_2,
            num_res_blocks,
            dropout_rate=dropout_rate,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
        )

        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.num_res_blocks = num_res_blocks
        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.epochs = epochs
        self.bs = batch_size
        self.lr = learning_rate
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.loss_func = loss_func
        self.device = device
        self.verbose = verbose

    def train(self, eval_every: Optional[int] = None, snapshot: Optional[int | list[int]] = None):
        '''Train the MLP.'''

        if isinstance(snapshot, int):
            snapshot = [snapshot]

        model = self.model.to(self.device)
        model.train()

        optimizer = self.optimizer(model.parameters(), self.lr, **self.optimizer_params)
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_params) if self.lr_scheduler is not None else None
        loss_func = self.loss_func()

        for epoch in range(self.epochs):
            # Generate a new train set each epoch
            data = DataLoader(self.data.generate(), self.bs, shuffle=True)
            for x, y in tqdm(data, desc=f'Training epoch {epoch + 1}/{self.epochs}', disable=not self.verbose):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                predictions = model(x).squeeze(-1)
                loss = loss_func(predictions, y)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if eval_every is not None and (epoch + 1) % eval_every == 0:
                self.eval(False)

            if (snapshot is not None and (epoch + 1) in snapshot) or epoch == self.epochs - 1:
                self.save(prefix=f'MLP_epoch{epoch + 1}_min')

        self.eval(True)

    def eval(self, plot: bool):
        '''Evaluate the MLP.'''

        model = self.model.to(self.device)
        model.eval()

        data = DataLoader(self.data.generate(), self.bs)
        predictions, targets = [], []
        with torch.no_grad():
            for x, y in tqdm(data, desc='Evaluating', disable=not self.verbose):
                x = x.to(self.device)
                prediction = model(x).squeeze(-1).cpu()
                predictions.append(prediction)
                targets.append(y)

        self.data.offload()
        model.train()  # Switch back to training mode before returning

        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

        if self.scale_data:
            predictions *= self.rw_length
            targets *= self.rw_length

        self.score(targets, predictions, plot)

    def search(self, start_cube: RubiksCube) -> list[str] | None:
        '''Search for a solution to a scrambled state.'''

        def score_fn(cubes: list[RubiksCube]) -> list[float]:
            cubes_tensor = RubiksCube._to_tensor([cube.state for cube in cubes], self.dtype, self.one_hot)
            return batch_infer(self.model.to(self.device), cubes_tensor, self.bs).tolist()

        return self._search(start_cube, score_fn)


class EmbeddingMLPAgent(Agent):
    '''A Rubik's cube agent using an Embedding MLP.'''

    def __init__(
        self,
        beam_size: int,
        max_search_its: int,
        rw_length: int,
        train_size: int,
        rw_accumulation_method: str,
        scale_data: bool,
        one_hot: bool,
        dtype: torch.dtype,
        target_dtype: torch.dtype,
        verbose: bool,
        # EmbeddingMLPAgent specific params:
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout_rate: float,
        activation_func: nn.Module,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optimizer: optim.Optimizer,
        optimizer_params: dict[str, Any],
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        lr_scheduler_params: Optional[dict[str, Any]],
        loss_func: nn.modules.loss._Loss,
        device: torch.device,
    ):
        '''
        Parameters
        ----------
        beam_size, max_search_its, rw_length, train_size, rw_accumulation_method,
        scale_data, one_hot, dtype, target_dtype, verbose
            See `Agent` class.
        embedding_dim, hidden_dim, n_layers, dropout_rate, activation_func
            See EmbeddingMLP class.
        epochs : int
            The number of epochs to train for.
        batch_size : int
            The batch size for model training.
        learning_rate : float
            The learning rate for model training.
        optimizer : optim.Optimizer
            The optimizer to optimize model parameters.
        optimizer_params : dict[str, Any]
            A dictionary of parameters for the optimizer, not including learning rate.
        lr_scheduler : Optional[optim.lr_scheduler.LRScheduler]
            Optionally, an LR scheduler.
        lr_scheduler_params : Optional[dict[str, Any]]
            A dictionary of parameters for the LR scheduler.
        loss_func : nn.modules.loss._Loss
            The loss function to train the model.
        device : torch.device
            The device to train the model on.
        '''

        super().__init__(
            beam_size=beam_size,
            max_search_its=max_search_its,
            rw_length=rw_length,
            train_size=train_size,
            rw_accumulation_method=rw_accumulation_method,
            scale_data=scale_data,
            one_hot=one_hot,
            dtype=dtype,
            target_dtype=target_dtype,
            verbose=verbose,
        )

        self.model = EmbeddingMLP(
            n_features=48,
            n_categories=6,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation_func=activation_func,
        )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.activation_func = activation_func
        self.epochs = epochs
        self.bs = batch_size
        self.lr = learning_rate
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.loss_func = loss_func
        self.device = device
        self.verbose = verbose

    def train(self, eval_every: Optional[int] = None, snapshot: Optional[int | list[int]] = None):
        '''Train the Embedding MLP.'''

        if isinstance(snapshot, int):
            snapshot = [snapshot]

        model = self.model.to(self.device)
        model.train()

        optimizer = self.optimizer(model.parameters(), self.lr, **self.optimizer_params)
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_params) if self.lr_scheduler is not None else None
        loss_func = self.loss_func()

        for epoch in range(self.epochs):
            # Generate a new train set each epoch
            data = DataLoader(self.data.generate(), self.bs, shuffle=True)
            for x, y in tqdm(data, desc=f'Training epoch {epoch + 1}/{self.epochs}', disable=not self.verbose):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                predictions = model(x).squeeze(-1)
                loss = loss_func(predictions, y)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if eval_every is not None and (epoch + 1) % eval_every == 0:
                self.eval(False)

            if (snapshot is not None and (epoch + 1) in snapshot) or epoch == self.epochs - 1:
                self.save(prefix=f'EmbeddingMLP_epoch{epoch + 1}_4l')

        self.eval(True)

    def eval(self, plot: bool):
        '''Evaluate the Embedding MLP.'''

        model = self.model.to(self.device)
        model.eval()

        data = DataLoader(self.data.generate(), self.bs, shuffle=False)
        predictions, targets = [], []
        with torch.no_grad():
            for x, y in tqdm(data, desc='Evaluating', disable=not self.verbose):
                x = x.to(self.device)
                prediction = model(x).squeeze(-1).cpu()
                predictions.append(prediction)
                targets.append(y)

        self.data.offload()
        model.train()  # Switch back to training mode before returning

        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

        if self.scale_data:
            predictions *= self.rw_length
            targets *= self.rw_length

        self.score(targets, predictions, plot)

    def search(self, start_cube: RubiksCube) -> list[str] | None:
        '''Search for a solution to the scrambled state.'''

        def score_fn(cubes: list[RubiksCube]) -> list[float]:
            cubes_tensor = RubiksCube._to_tensor([cube.state for cube in cubes], self.dtype, self.one_hot)
            return batch_infer(self.model.to(self.device), cubes_tensor, self.bs).tolist()

        return self._search(start_cube, score_fn)


class FTTransformerAgent(Agent):
    '''A Rubik's cube agent using an FT Transformer.'''

    def __init__(
        self,
        beam_size: int,
        max_search_its: int,
        rw_length: int,
        train_size: int,
        rw_accumulation_method: str,
        scale_data: bool,
        one_hot: bool,
        dtype: torch.dtype,
        target_dtype: torch.dtype,
        verbose: bool,
        # FTTransformerAgent specific params:
        d_model: int,
        n_heads: int,
        d_ff: int,
        depth: int,
        dropout_rate: float,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optimizer: optim.Optimizer,
        optimizer_params: dict[str, Any],
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        lr_scheduler_params: Optional[dict[str, Any]],
        loss_func: nn.modules.loss._Loss,
        device: torch.device,
    ):
        '''
        Parameters
        ----------
        beam_size, max_search_its, rw_length, train_size, rw_accumulation_method,
        scale_data, one_hot, dtype, target_dtype, verbose
            See `Agent` class.
        d_model, n_heads, d_ff, depth, dropout_rate
            See FTTransformer class.
        epochs : int
            The number of epochs to train for.
        batch_size : int
            The batch size for model training.
        learning_rate : float
            The learning rate for model training.
        optimizer : optim.Optimizer
            The optimizer to optimize model parameters.
        optimizer_params : dict[str, Any]
            A dictionary of parameters for the optimizer, not including learning rate.
        lr_scheduler : Optional[optim.lr_scheduler.LRScheduler]
            Optionally, an LR scheduler.
        lr_scheduler_params : Optional[dict[str, Any]]
            A dictionary of parameters for the LR scheduler.
        loss_func : nn.modules.loss._Loss
            The loss function to train the model.
        device : torch.device
            The device to train the model on.
        '''

        super().__init__(
            beam_size=beam_size,
            max_search_its=max_search_its,
            rw_length=rw_length,
            train_size=train_size,
            rw_accumulation_method=rw_accumulation_method,
            scale_data=scale_data,
            one_hot=one_hot,
            dtype=dtype,
            target_dtype=target_dtype,
            verbose=verbose,
        )

        self.model = FTTransformer(
            n_features=48,
            n_classes=6,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            depth=depth,
            dropout_rate=dropout_rate,
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.bs = batch_size
        self.lr = learning_rate
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.loss_func = loss_func
        self.device = device
        self.verbose = verbose

    def train(self, eval_every: Optional[int] = None, snapshot: Optional[int | list[int]] = None):
        '''Train the FT Transformer.'''

        if isinstance(snapshot, int):
            snapshot = [snapshot]

        model = self.model.to(self.device)
        model.train()

        optimizer = self.optimizer(model.parameters(), self.lr, **self.optimizer_params)
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_params) if self.lr_scheduler is not None else None
        loss_func = self.loss_func()

        for epoch in range(self.epochs):
            # Generate a new train set each epoch
            data = DataLoader(self.data.generate(), self.bs, shuffle=True)
            for x, y in tqdm(data, desc=f'Training epoch {epoch + 1}/{self.epochs}', disable=not self.verbose):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                predictions = model(x).squeeze(-1)
                loss = loss_func(predictions, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if eval_every is not None and (epoch + 1) % eval_every == 0:
                self.eval(False)

            if (snapshot is not None and (epoch + 1) in snapshot) or epoch == self.epochs - 1:
                self.save(prefix=f'FTTransformer_epoch{epoch + 1}')

        self.eval(True)

    def eval(self, plot: bool):
        '''Evaluate the FT Transformer.'''

        model = self.model.to(self.device)
        model.eval()

        data = DataLoader(self.data.generate(), self.bs, shuffle=False)
        predictions, targets = [], []
        with torch.no_grad():
            for x, y in tqdm(data, desc='Evaluating', disable=not self.verbose):
                x = x.to(self.device)
                prediction = model(x).squeeze(-1).cpu()
                predictions.append(prediction)
                targets.append(y)

        self.data.offload()
        model.train()  # Switch back to training mode before returning

        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

        if self.scale_data:
            predictions *= self.rw_length
            targets *= self.rw_length

        self.score(targets, predictions, plot)

    def search(self, start_cube: RubiksCube) -> list[str] | None:
        '''Search for a solution to the unsolved cube.'''

        def score_fn(cubes: list[RubiksCube]) -> list[float]:
            cubes_tensor = RubiksCube._to_tensor([cube.state for cube in cubes], self.dtype, self.one_hot)
            return batch_infer(self.model.to(self.device), cubes_tensor, self.bs).tolist()

        return self._search(start_cube, score_fn)


class RankAgent(Agent):
    '''A Rubik's cube agent using a rank model.'''

    def __init__(
        self,
        beam_size: int,
        max_search_its: int,
        rw_length: int,
        train_size: int,
        batch_generation_size: int,
        rw_accumulation_method: str,
        scale_data: bool,
        one_hot: bool,
        dtype: torch.dtype,
        target_dtype: torch.dtype,
        verbose: bool,
        # RankAgent specific params:
        cat_embed_dim: int,
        hidden_dim: int,
        embed_dim: int,
        n_layers: int,
        learning_rate: float,
        batch_size: int,
        update_every_n_batches: int,
        optimizer: optim.Optimizer,
        optimizer_params: dict[str, Any],
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        lr_scheduler_params: Optional[dict[str, Any]],
        eval_size: int,
        device: torch.device,
    ):
        '''
        Parameters
        ----------
        beam_size, max_search_its, rw_length, train_size, rw_accumulation_method,
        scale_data, one_hot, dtype, target_dtype, verbose
            See `Agent` class.
        cat_embed_dim, hidden_dim, embed_dim, n_layers
            See MLPEncoder and RankModel classes.
        learning_rate : float
            The learning rate for model training.
        batch_size : int
            The batch size for model training.
        update_every_n_batches : int
            The update interval for the LR scheduler and model evaluation.
        optimizer : optim.Optimizer
            The optimizer to optimize model parameters.
        optimizer_params : dict[str, Any]
            A dictionary of parameters for the optimizer, not including learning rate.
        lr_scheduler : Optional[optim.lr_scheduler.LRScheduler]
            Optionally, an LR scheduler.
        lr_scheduler_params : Optional[dict[str, Any]]
            A dictionary of parameters for the LR scheduler.
        eval_size : int
            The number of samples to evaluate on.
        device : torch.device
            The device to train the model on.
        '''

        super().__init__(
            beam_size=beam_size,
            max_search_its=max_search_its,
            rw_length=rw_length,
            train_size=train_size,
            rw_accumulation_method=rw_accumulation_method,
            scale_data=scale_data,
            one_hot=one_hot,
            dtype=dtype,
            target_dtype=target_dtype,
            verbose=verbose,
        )

        self.batch_generation_size = batch_generation_size
        self.data = PairwiseRubiksCubeDataset(
            k=rw_length,
            total=train_size,
            batch_generation_size=batch_generation_size,
            method=rw_accumulation_method,
            dtype=dtype,
            target_dtype=target_dtype,
            scale=scale_data,
            one_hot=one_hot,
            shuffle=True,
            verbose=verbose,
        )  # Overwrites `self.data` from the superclass

        self.encoder = MLPEncoder(
            n_features=48,
            n_categories=6,
            cat_embed_dim=cat_embed_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            n_layers=n_layers,
        )
        self.model = RankModel(encoder=self.encoder, anchor_state=RubiksCube().to_tensor(self.dtype, one_hot))

        self.cat_embed_dim = cat_embed_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.update_every_n_batches = update_every_n_batches
        self.lr = learning_rate
        self.bs = batch_size
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.device = device
        self.eval_size = eval_size

    def train(self, snapshot: list[int] = []):
        '''Train the rank model.'''

        model = self.model.to(self.device)
        model.train()

        data = DataLoader(self.data, self.bs)

        optimizer = self.optimizer(model.parameters(), self.lr, **self.optimizer_params)
        if isinstance(self.lr_scheduler_params, MutableMapping) and 'T_max' not in self.lr_scheduler_params:
            self.lr_scheduler_params['T_max'] = len(data) // self.update_every_n_batches
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_params) if self.lr_scheduler is not None else None

        samples_trained = 0
        for i, ((closer, _), (farther, _)) in enumerate(data):
            closer, farther = closer.to(self.device), farther.to(self.device)

            optimizer.zero_grad()
            loss = model(closer, farther)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            samples_trained += closer.shape[0]

            if self.verbose:
                fields = [f'It: {i + 1:_}/{self.train_size // self.bs:_}']
                if scheduler is not None:
                    fields.append(f'LR: {scheduler.get_last_lr()}')
                fields.append(f'Loss: {loss.item():.4f}')
                print('  |  '.join(fields))

            if (i + 1) % self.update_every_n_batches == 0:
                if scheduler is not None:
                    scheduler.step()
                if self.verbose:
                    self.eval()

            if (i + 1) in snapshot or i == len(data) - 1:
                self.save(prefix=f'Rank_batch{i + 1}')

    def eval(self):
        '''Evaluate the rank model.'''

        model = self.model.to(self.device)
        model.eval()

        n_correct = n_total = 0
        data = DataLoader(self.data, self.bs)
        with torch.no_grad():
            for (closer, _), (farther, _) in data:
                closer, farther = closer.to(self.device), farther.to(self.device)
                d_closer = model.distance(closer)
                d_farther = model.distance(farther)
                n_correct += (d_closer < d_farther).sum().item()
                n_total += closer.shape[0]
                if n_total >= self.eval_size:
                    break

        model.train()  # Switch back to training mode before returning
        print(f'Accuracy: {n_correct / n_total:.2%}')

    def search(self, start_cube: RubiksCube) -> list[str] | None:
        '''Search for a solution to a scrambled state.'''

        def score_fn(cubes: list[RubiksCube]) -> list[float]:
            cubes_tensor = RubiksCube._to_tensor([cube.state for cube in cubes], self.dtype, self.one_hot)
            loader = DataLoader(TensorDataset(cubes_tensor), self.bs, shuffle=False)

            distances = []
            model = self.model.to(self.device)
            with torch.no_grad():
                for (batch,) in loader:
                    batch = batch.to(self.device)
                    distances.append(model.distance(batch).flatten().cpu() + 1)
            distances = torch.cat(distances, dim=0)

            return distances.tolist()

        return self._search(start_cube, score_fn)


class MLXEmbeddingMLPAgent(Agent):
    '''The is an implementation of the EmbeddingMLP agent in MLX.'''

    def __init__(
        self,
        beam_size: int,
        max_search_its: int,
        rw_length: int,
        train_size: int,
        rw_accumulation_method: str,
        scale_data: bool,
        one_hot: bool,
        dtype: np.dtype,
        target_dtype: np.dtype,
        verbose: bool,
        # MLXEmbeddingMLPAgent specific params:
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout_rate: float,
        activation_func: mlx_nn.Module,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optimizer: mlx_optim.Optimizer,
        optimizer_params: dict[str, Any],
        lr_scheduler: Optional[Callable[[int], float]],
        loss_func: Callable[[mx.array, mx.array], mx.array],
    ):
        '''
        Parameters
        ----------
        beam_size, max_search_its, rw_length, train_size, rw_accumulation_method,
        scale_data, one_hot, dtype, target_dtype, verbose
            See `Agent` class.
        embedding_dim, hidden_dim, n_layers, dropout_rate, activation_func
            See EmbeddingMLP class.
        epochs : int
            The number of epochs to train for.
        batch_size : int
            The batch size for model training.
        learning_rate : float
            The learning rate for model training.
        optimizer : optim.Optimizer
            The optimizer to optimize model parameters.
        optimizer_params : dict[str, Any]
            A dictionary of parameters for the optimizer, not including learning rate.
        lr_scheduler : Optional[optim.lr_scheduler.LRScheduler]
            Optionally, an LR scheduler.
        lr_scheduler_params : Optional[dict[str, Any]]
            A dictionary of parameters for the LR scheduler.
        loss_func : nn.modules.loss._Loss
            The loss function to train the model.
        '''

        super().__init__(
            beam_size=beam_size,
            max_search_its=max_search_its,
            rw_length=rw_length,
            train_size=train_size,
            rw_accumulation_method=rw_accumulation_method,
            scale_data=scale_data,
            one_hot=one_hot,
            dtype=dtype,
            target_dtype=target_dtype,
            verbose=verbose,
        )

        self.model = MLXEmbeddingMLP(
            n_features=48,
            n_categories=6,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation_func=activation_func,
        )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.activation_func = activation_func
        self.epochs = epochs
        self.bs = batch_size
        self.lr = learning_rate
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.loss_func = loss_func
        self.verbose = verbose

    def batch_mlx_dataset(self, dataset: RubiksCubeData, shuffle: bool = True) -> list[tuple[mx.array, mx.array]]:
        '''Convert the dataset into batches of MLX arrays.'''

        if self.verbose:
            print('Batching dataset...')

        indices = list(range(len(dataset)))
        if shuffle:
            import random

            random.shuffle(indices)

        batches = []
        end = 0
        while end < len(dataset):
            start = end
            end += self.bs
            arrays = [dataset[i] for i in indices[start:end]]
            x, y = zip(*arrays)
            x = mx.array(np.stack(x))
            y = mx.array(np.stack(y))
            batches.append((x, y))

        return batches

    def train(self, eval_every: Optional[int] = None, snapshot: Optional[int | list[int]] = None):
        '''Train the Embedding MLP using MLX.'''

        if isinstance(snapshot, int):
            snapshot = [snapshot]

        self.model.train()

        optimizer: mlx_optim.Optimizer = self.optimizer(learning_rate=self.lr, **self.optimizer_params)

        def loss_fn(model: mlx_nn.Module, xb: mx.array, yb: mx.array) -> tuple[mx.array, mx.array]:
            return self.loss_func(model(xb).squeeze(-1), yb)

        state = [self.model.state, optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(x: mx.array, y: mx.array):
            loss_and_grad = mlx_nn.value_and_grad(self.model, loss_fn)
            loss, grads = loss_and_grad(self.model, x, y)
            mlx_optim.clip_grad_norm(grads, max_norm=2.0)
            if self.lr_scheduler is not None:
                optimizer.learning_rate = self.lr_scheduler(epoch)
            optimizer.update(self.model, grads)
            return loss

        for epoch in range(self.epochs):
            # Generate a new train set each epoch
            data = self.batch_mlx_dataset(self.data.generate())
            self.data.offload()
            for x, y in tqdm(data, desc=f'Training epoch {epoch + 1}/{self.epochs}', disable=not self.verbose):
                loss = step(x, y)
                mx.eval(state)

            del data

            if eval_every is not None and (epoch + 1) % eval_every == 0:
                self.eval(False)

            if (snapshot is not None and (epoch + 1) in snapshot) or epoch == self.epochs - 1:
                self.save(prefix=f'MLXEmbeddingMLP_epoch{epoch + 1}')

        self.eval(True)

    def eval(self, plot: bool):
        '''Evaluate the Embedding MLP.'''

        self.model.eval()

        data = self.batch_mlx_dataset(self.data.generate(), False)
        self.data.offload()
        predictions, targets = [], []
        for x, y in tqdm(data, desc='Evaluating', disable=not self.verbose):
            prediction = self.model(x).squeeze(-1)
            predictions.append(np.array(prediction))
            targets.append(np.array(y))

        self.model.train()  # Switch back to training mode before returning

        predictions = np.concat(predictions, axis=0)
        targets = np.concat(targets, axis=0)

        if self.scale_data:
            predictions *= self.rw_length
            targets *= self.rw_length

        self.score(targets, predictions, plot)

    def search(self, start_cube: RubiksCube) -> list[str] | None:
        '''Search for a solution to the scrambled state.'''

        def score_fn(cubes: list[RubiksCube]) -> list[float]:
            cubes_array = RubiksCube._to_ndarray((cube.state for cube in cubes), self.dtype, self.one_hot)
            return batch_infer_mlx(self.model, cubes_array, self.bs).tolist()

        return self._search(start_cube, score_fn)


def test_agent(path: Path | str):
    '''Test an agent through varying beam sizes.'''

    import statistics as stats

    beam_sizes = [2**6, 2**8, 2**10, 2**11, 2**12, 2**13, 2**14]
    trials = 400

    agent = Agent.from_path(path)
    # agent.verbose = False

    results = {}
    for beam_size in beam_sizes:
        print(f'Beam size: {beam_size}')
        agent.beam_size = beam_size

        successes = []
        for _ in range(trials):
            cube = RubiksCube().scramble(50)  # Deep scramble
            solution = agent.search(cube)
            if solution is not None:
                successes.append(len(solution))
        results[beam_size] = {
            'successes': successes,
            'n_successes': len(successes),
            'avg_sol_len': sum(successes) / len(successes) if successes else -1,
            'median_sol_len': stats.median(successes) if successes else -1,
            'stdev_sol_len': stats.stdev(successes) if len(successes) >= 2 else -1,
        }

    print(f'Agent: {path}')
    for beam_size in beam_sizes:
        res = results[beam_size]
        print(f'Beam size: {beam_size}')
        print(f'\tSuccesses:                          {res['n_successes']:_}/{trials:_}')
        print(f'\tMean solution length:               {res['avg_sol_len']:.4f}')
        print(f'\tMedian solution length:             {res['median_sol_len']:.4f}')
        print(f'\tSolution length standard deviation: {res['stdev_sol_len']:.4f}')
    print()


if __name__ == '__main__':
    # epochs = 2000
    # agent = MLPAgent(
    #     beam_size=2**12,
    #     max_search_its=50,
    #     rw_length=26,
    #     rw_accumulation_method='min',
    #     scale_data=True,
    #     one_hot=True,
    #     dtype=torch.float32,
    #     target_dtype=torch.float32,
    #     hidden_dim_1=900,
    #     hidden_dim_2=628,
    #     num_res_blocks=4,
    #     activation_func=nn.ReLU,
    #     dropout_rate=0.1,
    #     use_batch_norm=True,
    #     train_size=1_000_000,
    #     epochs=epochs,
    #     batch_size=10_000,
    #     learning_rate=1.5e-3,
    #     optimizer=optim.AdamW,
    #     optimizer_params={'weight_decay': 1e-5},
    #     lr_scheduler=optim.lr_scheduler.CosineAnnealingLR,
    #     lr_scheduler_params={'T_max': epochs, 'eta_min': 1e-4},
    #     loss_func=nn.SmoothL1Loss,
    #     device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
    #     verbose=True,
    # )
    # agent.train(eval_every=50, snapshot=[250, 500, 750, 1000, 1250, 1500, 1750, 2000])

    # epochs = 2000
    # agent = EmbeddingMLPAgent(
    #     beam_size=2**12,
    #     max_search_its=50,
    #     rw_length=26,
    #     train_size=1_000_000,
    #     rw_accumulation_method='mean',
    #     scale_data=True,
    #     one_hot=False,
    #     dtype=torch.long,
    #     target_dtype=torch.float32,
    #     verbose=True,
    #     embedding_dim=16,
    #     hidden_dim=768,
    #     # n_layers=6,
    #     n_layers=4,
    #     dropout_rate=0.1,
    #     activation_func=nn.GELU,
    #     epochs=epochs,
    #     batch_size=10_000,
    #     learning_rate=1.5e-3,
    #     optimizer=optim.AdamW,
    #     optimizer_params={'weight_decay': 1e-5},
    #     lr_scheduler=optim.lr_scheduler.CosineAnnealingLR,
    #     lr_scheduler_params={'T_max': epochs, 'eta_min': 1e-4},
    #     loss_func=nn.SmoothL1Loss,
    #     device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
    # )
    # agent.train(eval_every=100, snapshot=[1750, 2000])

    # epochs = 1000
    # agent = FTTransformerAgent(
    #     beam_size=2**12,
    #     max_search_its=50,
    #     rw_length=26,
    #     train_size=1_000_000,
    #     rw_accumulation_method='mean',
    #     scale_data=True,
    #     one_hot=False,
    #     dtype=torch.long,
    #     target_dtype=torch.float32,
    #     verbose=True,
    #     d_model=64,
    #     n_heads=4,
    #     d_ff=128,
    #     depth=4,
    #     dropout_rate=0.1,
    #     epochs=epochs,
    #     batch_size=4_000,
    #     learning_rate=1e-3,
    #     optimizer=optim.AdamW,
    #     optimizer_params={'weight_decay': 1e-5},
    #     lr_scheduler=optim.lr_scheduler.CosineAnnealingLR,
    #     lr_scheduler_params={'T_max': epochs, 'eta_min': 1e-4},
    #     loss_func=nn.SmoothL1Loss,
    #     device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
    # )
    # agent.train(eval_every=50, snapshot=[250, 500, 750, 1000])

    # agent = RankAgent(
    #     beam_size=2**12,
    #     max_search_its=50,
    #     rw_length=26,
    #     train_size=2_000_000_000,
    #     batch_generation_size=1_000_000,
    #     rw_accumulation_method='mean',
    #     scale_data=False,
    #     one_hot=False,
    #     dtype=torch.long,
    #     target_dtype=torch.float32,
    #     verbose=True,
    #     cat_embed_dim=16,
    #     hidden_dim=512,
    #     embed_dim=128,
    #     n_layers=6,
    #     learning_rate=1.5e-3,
    #     batch_size=10_000,
    #     update_every_n_batches=100,
    #     optimizer=optim.AdamW,
    #     optimizer_params={'weight_decay': 1e-5},
    #     lr_scheduler=optim.lr_scheduler.CosineAnnealingLR,
    #     lr_scheduler_params={'eta_min': 3e-5},
    #     eval_size=100_000,
    #     device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'),
    # )
    # agent.train(snapshot=[25_000, 50_000, 100_000, 150_000, 200_000])

    # epochs = 2000
    # agent = MLXEmbeddingMLPAgent(
    #     beam_size=2**12,
    #     max_search_its=50,
    #     rw_length=26,
    #     train_size=1_000_000,
    #     rw_accumulation_method='mean',
    #     scale_data=True,
    #     one_hot=False,
    #     dtype=np.int64,
    #     target_dtype=np.float32,
    #     verbose=True,
    #     embedding_dim=16,
    #     hidden_dim=768,
    #     n_layers=6,
    #     dropout_rate=0.1,
    #     activation_func=mlx_nn.GELU,
    #     epochs=epochs,
    #     batch_size=10_000,
    #     learning_rate=1e-3,
    #     optimizer=mlx_optim.Adam,
    #     optimizer_params={},  # {'weight_decay': 1e-5},
    #     lr_scheduler=None,
    #     loss_func=mlx_nn.losses.smooth_l1_loss,
    # )
    # agent.train(eval_every=50, snapshot=[250, 500, 750, 1000, 1250, 1500, 1750, 2000])

    agents = [
        './agent_cache/MLP_epoch2000_5660508080.pkl',
        './agent_cache/MLP_epoch2000_min_5889334240.pkl',
        './agent_cache/EmbeddingMLP_epoch2000_5368093568.pkl',
        './agent_cache/EmbeddingMLP_epoch2000_min_6085569248.pkl',
        './agent_cache/Rank_batch200000_5818525376.pkl',
        './agent_cache/EmbeddingMLP_epoch2000_k20_4740610480.pkl',
        './agent_cache/EmbeddingMLP_epoch2000_5l_5247859456.pkl',
        './agent_cache/EmbeddingMLP_epoch2000_4l_4769708848.pkl',
    ]

    test_agent(agents[7])
