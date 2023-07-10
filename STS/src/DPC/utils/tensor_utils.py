import torch
import numbers
import numpy as np 
from scipy.spatial import cKDTree

def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor,numbers.Number):
        return np.array(tensor)
    else:
        raise NotImplementedError

def to_tensor(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray
    elif isinstance(ndarray, np.ndarray):
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray,numbers.Number):
        return torch.tensor(ndarray)
    else:
        raise NotImplementedError

def to_number(ndarray):
    if isinstance(ndarray, torch.Tensor) or isinstance(ndarray, np.ndarray):
        return ndarray.item()
    elif isinstance(ndarray,numbers.Number):
        return ndarray
    else:
        raise NotImplementedError

def mean_non_zero(tensor,axis):
    mask = tensor!=0
    tensor_mean = (tensor*mask).sum(dim=axis)/mask.sum(dim=axis)
    return tensor_mean


def recursive_dict_tensor_to_cuda(dict_of_tensors: dict, subset_batch=None):
    for k in dict_of_tensors.keys():
        if isinstance(dict_of_tensors[k], dict):
            recursive_dict_tensor_to_cuda(dict_of_tensors[k], subset_batch)
        else:
            if(isinstance(dict_of_tensors[k], torch.Tensor)):
                dict_of_tensors[k] = dict_of_tensors[k].cuda()
                if(subset_batch):
                    dict_of_tensors[k]=dict_of_tensors[k][:subset_batch]
    return dict_of_tensors

# source: https://pytorch-geometric.readthedocs.io/en/1.3.0/_modules/torch_cluster/knn.html
# slight changes from source code
def knn(x, y, k, batch_x=None, batch_y=None):
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import knn

    .. testcode::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch_x = torch.tensor([0, 0, 0, 0])
        >>> y = torch.Tensor([[-1, 0], [1, 0]])
        >>> batch_x = torch.tensor([0, 0])
        >>> assign_index = knn(x, y, 2, batch_x, batch_y)
    """

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    # Rescale x and y.
    min_xy = min(x.min().item(), y.min().item())
    x, y = x - min_xy, y - min_xy

    max_xy = max(x.max().item(), y.max().item())
    x, y, = x / max_xy, y / max_xy

    # Concat batch/features to ensure no cross-links between examples exist.
    x = torch.cat([x, 2 * x.size(1) * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * y.size(1) * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    tree = cKDTree(x.detach().cpu().numpy())
    dist, col = tree.query(y.detach().cpu(), k=k, distance_upper_bound=x.size(1))
    dist = torch.from_numpy(dist).to(x.dtype)
    col = torch.from_numpy(col).to(torch.long)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
    mask = ~torch.isinf(dist).view(-1)
    row, col = row.view(-1)[mask], col.view(-1)[mask]

    return torch.stack([row, col], dim=0).to(x.device)