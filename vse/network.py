"""Single-hidden-layer network representation and structural operators.

A *chromosome* is a single-hidden-layer MLP whose hidden-layer size is allowed
to vary (this is the "variable structure encoding", VSE).  It is stored as a
small object with four arrays:

    hidden_layer.weight : (hidden_size, input_size)
    hidden_layer.bias   : (hidden_size, 1)
    output_layer.weight : (output_size, hidden_size)
    output_layer.bias   : (output_size, 1)

The functions here are faithful ports of the MATLAB helpers
``InitialSingleLayerNetwork`` (reconstructed -- see note below),
``NetworkToVector``, ``VectorToNetwork`` and the
Addition/Elimination/Substitution/Mutation/Crossover operators.

NOTE on ``InitialSingleLayerNetwork``: the MATLAB function was referenced from
an external path and is not present in the original ``experiment_codes`` folder.
Its behaviour is reconstructed unambiguously from how every operator builds new
weights, i.e. ``lb + (ub - lb) * rand`` for hidden weights/biases and output
weights/biases.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Layer:
    weight: np.ndarray
    bias: np.ndarray


@dataclass
class Network:
    hidden_layer: Layer
    output_layer: Layer

    def copy(self) -> "Network":
        return Network(
            Layer(self.hidden_layer.weight.copy(), self.hidden_layer.bias.copy()),
            Layer(self.output_layer.weight.copy(), self.output_layer.bias.copy()),
        )

    @property
    def hidden_size(self) -> int:
        return self.hidden_layer.weight.shape[0]

    @property
    def input_size(self) -> int:
        return self.hidden_layer.weight.shape[1]

    @property
    def output_size(self) -> int:
        return self.output_layer.weight.shape[0]


# --------------------------------------------------------------------------- #
# Construction                                                                 #
# --------------------------------------------------------------------------- #
def initial_single_layer_network(opts: dict, rng: np.random.Generator) -> Network:
    """Reconstructed ``InitialSingleLayerNetwork``.

    ``opts`` must provide ``inp``, ``hidn``, ``outp``, ``lb`` and ``ub``.
    """
    inp = int(opts["inp"])
    hidn = int(opts["hidn"])
    outp = int(opts["outp"])
    lb = float(opts["lb"])
    ub = float(opts["ub"])
    span = ub - lb
    return Network(
        Layer(
            lb + span * rng.random((hidn, inp)),
            lb + span * rng.random((hidn, 1)),
        ),
        Layer(
            lb + span * rng.random((outp, hidn)),
            lb + span * rng.random((outp, 1)),
        ),
    )


# --------------------------------------------------------------------------- #
# Vectorisation                                                                #
# --------------------------------------------------------------------------- #
def network_to_vector(net: Network):
    """Flatten a network to a 1-D vector (matching MATLAB ``NetworkToVector``).

    Layout: for each hidden node ``[w_1..w_inp, bias]`` then for each output
    node ``[w_1..w_hidn, bias]``.

    Returns ``(vector, meta)`` where ``meta = (inp, hidn, outp)`` is the
    structural information needed to rebuild the network.
    """
    hw, hb = net.hidden_layer.weight, net.hidden_layer.bias
    ow, ob = net.output_layer.weight, net.output_layer.bias
    hidn, inp = hw.shape
    outp = ow.shape[0]
    hidden_block = np.concatenate([hw, hb], axis=1).reshape(-1)   # (hidn*(inp+1),)
    output_block = np.concatenate([ow, ob], axis=1).reshape(-1)   # (outp*(hidn+1),)
    vec = np.concatenate([hidden_block, output_block])
    return vec, (inp, hidn, outp)


def vector_to_network(vec: np.ndarray, meta) -> Network:
    """Inverse of :func:`network_to_vector` (matching ``VectorToNetwork``)."""
    inp, hidn, outp = meta
    vec = np.asarray(vec, dtype=float)
    n_hidden = hidn * (inp + 1)
    hidden_block = vec[:n_hidden].reshape(hidn, inp + 1)
    output_block = vec[n_hidden:n_hidden + outp * (hidn + 1)].reshape(outp, hidn + 1)
    hw = hidden_block[:, :inp].copy()
    hb = hidden_block[:, inp:inp + 1].copy()
    ow = output_block[:, :hidn].copy()
    ob = output_block[:, hidn:hidn + 1].copy()
    return Network(Layer(hw, hb), Layer(ow, ob))


# --------------------------------------------------------------------------- #
# Structural operators                                                         #
# --------------------------------------------------------------------------- #
def addition_operator(net: Network, opts: dict, rng: np.random.Generator) -> Network:
    """Add one hidden node (port of ``AdditionOperator``)."""
    out = net.copy()
    inp = out.input_size
    hidden_size = out.hidden_size
    output_size = out.output_size
    lb, ub = float(opts["lb"]), float(opts["ub"])
    span = ub - lb
    if hidden_size < int(opts["max_hidden_size"]):
        new_row = lb + span * rng.random((1, inp))
        new_bias = lb + span * rng.random((1, 1))
        out.hidden_layer.weight = np.vstack([out.hidden_layer.weight, new_row])
        out.hidden_layer.bias = np.vstack([out.hidden_layer.bias, new_bias])
        new_col = lb + span * rng.random((output_size, 1))
        out.output_layer.weight = np.hstack([out.output_layer.weight, new_col])
    return out


def elimination_operator(net: Network, opts: dict, rng: np.random.Generator) -> Network:
    """Remove one hidden node (port of ``EliminationOperator``)."""
    out = net.copy()
    hidden_size = out.hidden_size
    if hidden_size > int(opts["min_hidden_size"]):
        # MATLAB: selected = round(hidden_size * rand + 0.5) in [1, hidden_size]
        selected = int(np.floor(hidden_size * rng.random() + 0.5))
        selected = min(max(selected, 1), hidden_size)
        keep = [i for i in range(hidden_size) if i != (selected - 1)]
        out.hidden_layer.weight = out.hidden_layer.weight[keep, :].copy()
        out.hidden_layer.bias = out.hidden_layer.bias[keep, :].copy()
        out.output_layer.weight = out.output_layer.weight[:, keep].copy()
    return out


def substitution_operator(net: Network, opts: dict, rng: np.random.Generator) -> Network:
    """Re-initialise one hidden node (port of ``SubstitutionOperator``)."""
    out = net.copy()
    inp = out.input_size
    hidden_size = out.hidden_size
    lb, ub = float(opts["lb"]), float(opts["ub"])
    span = ub - lb
    selected = int(np.floor(hidden_size * rng.random() + 0.5))
    selected = min(max(selected, 1), hidden_size) - 1
    out.hidden_layer.weight[selected, :] = lb + span * rng.random(inp)
    out.hidden_layer.bias[selected, 0] = lb + span * rng.random()
    return out


def mutation_operator(net: Network, opts: dict, probability: float,
                      rng: np.random.Generator) -> Network:
    """Per-gene uniform perturbation (port of ``MutationOperator``)."""
    vec, meta = network_to_vector(net)
    mask = rng.random(vec.shape) <= probability
    perturb = (0.2 * rng.random(vec.shape) - 0.1)  # UniformRandomNumber(-0.1, 0.1)
    vec = vec + mask * perturb
    return vector_to_network(vec, meta)


def crossover_operator(pop, rng: np.random.Generator):
    """In-place fitness-weighted 2-point hidden-node crossover.

    Port of ``CrossoverOperator``.  ``pop`` is a list of individuals each with
    ``.chrom`` (Network) and ``.fitness``; two parents are chosen by roulette
    wheel on inverse fitness and swap up to 2 hidden nodes.  Returns ``pop``.
    """
    fitnesses = np.array([ind.fitness for ind in pop], dtype=float)
    inv = 1.0 / fitnesses
    cum_sum = np.cumsum(inv / inv.sum())
    d = rng.random(2)
    c1 = int(np.searchsorted(cum_sum, d[0], side="right"))
    c2 = int(np.searchsorted(cum_sum, d[1], side="right"))
    c1 = min(c1, len(pop) - 1)
    c2 = min(c2, len(pop) - 1)

    n1 = pop[c1].chrom
    n2 = pop[c2].chrom
    max_h = min(n1.hidden_size, n2.hidden_size)
    if max_h < 1:
        return pop
    crossing_points = min(2, max_h)
    # MATLAB randi([1, max_h], 1, crossing_points)
    pts = rng.integers(1, max_h + 1, size=crossing_points)
    out_size = n1.output_size
    for p in pts:
        idx = p - 1
        aw = n1.hidden_layer.weight[idx, :].copy()
        ab = n1.hidden_layer.bias[idx, 0]
        bw = n2.hidden_layer.weight[idx, :].copy()
        bb = n2.hidden_layer.bias[idx, 0]
        n1.hidden_layer.weight[idx, :] = bw
        n1.hidden_layer.bias[idx, 0] = bb
        n2.hidden_layer.weight[idx, :] = aw
        n2.hidden_layer.bias[idx, 0] = ab
        for o in range(out_size):
            oaw = n1.output_layer.weight[o, idx]
            obw = n2.output_layer.weight[o, idx]
            n1.output_layer.weight[o, idx] = obw
            n2.output_layer.weight[o, idx] = oaw
    return pop


def random_hidden_size(opts: dict, rng: np.random.Generator) -> int:
    """``min + floor((max - min) * rand)`` -- the MATLAB VSE size draw."""
    lo = int(opts["min_hidden_size"])
    hi = int(opts["max_hidden_size"])
    return lo + int(np.floor((hi - lo) * rng.random()))
