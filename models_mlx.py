import mlx.core as mx
import mlx.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class MLXEmbeddingMLP(nn.Module):
    '''A multi-layer perceptron implemented with MLX that embeds each categorical feature into
    a dense vector of size `embedding_dim`, flattens the embeddings, and processes them
    through several fully connected layers to produce a scalar output.
    '''

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
            The number of categorical features in the dataset.
        n_categories : int
            The number of categories for each feature.
        embedding_dim : int
            The dimension of the embedding vectors.
        hidden_dim : int
            The hidden dimension of the fully connected layers.
        n_layers : int
            The number of hidden layers in the MLP.
        dropout_rate : float
            The dropout probability applied after each hidden layer.
        activation_func : nn.Module
            The activation function class to use (e.g., nn.ReLU).
        '''

        super().__init__()

        self.n_features = n_features
        self.n_categories = n_categories

        self.embeddings = nn.Embedding(n_features * n_categories, embedding_dim)
        self.offsets = mx.arange(n_features) * n_categories

        in_dim = embedding_dim * n_features

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(activation_func())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.embeddings(x + self.offsets.astype(x.dtype))  # shape: (batch, n_features, embedding_dim)
        x = x.reshape(x.shape[0], -1)
        return mx.sigmoid(self.mlp(x))


def batch_infer_mlx(model: MLXEmbeddingMLP, x: np.ndarray, batch_size: int) -> np.ndarray:
    '''Infer the MLP model in batches.

    Parameters
    ----------
    model : MLP
        An MLP model to perform inference on.
    x : np.ndarray
        An input tensor.
    batch_size : int
        The batch size to use.

    Returns
    -------
    np.ndarray
        The result of applying the model to the input array.
    '''

    loader = DataLoader(TensorDataset(torch.tensor(x)), batch_size=batch_size, shuffle=False)
    model.eval()

    outputs = []
    for (batch,) in loader:
        outputs.append(np.array(model(mx.array(batch.numpy())).squeeze(-1)))
    return np.concat(outputs, axis=0)
