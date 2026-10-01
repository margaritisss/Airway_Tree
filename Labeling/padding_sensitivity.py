"""Measure how much IPGN's voxel labels shift when the graph node-padding changes.

Runs the same case three ways and compares predicted labels voxel-by-voxel:
  A) node_pad=519, point-cloud seed 0   <- the training-time padding
  B) node_pad=850, point-cloud seed 0   <- only the padding differs from A
  C) node_pad=519, point-cloud seed 1   <- only the sampling differs from A

A-vs-B is the padding effect. A-vs-C is the noise floor from random point
sampling. The padding only matters if A-vs-B is clearly larger than A-vs-C.
"""

import argparse                                        # CLI so paths aren't hardcoded
import os                                              # path joins for specs/checkpoint
import json                                            # specs files are JSON
import numpy as np                                     # volume + point handling
import networkx as nx                                  # graphml reader
import torch                                           # inference

from torch_geometric.data import Data                  # minmax_norm_g expects this shape of object
import model.data_augmentation as aug                  # reuse the repo's exact normalization
import model.pg_model as pg_model                      # the IPGN network

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # repo root, for specs and ckpt
TOTAL_SAMPLE_POINTS = 6000                             # point-cloud size, as in pygeo_dataset.py
IMPLICIT_POINTS = 10000                                # query points, as in pygeo_dataset.py


def build_sample(npz_path, gml_path, node_pad, point_seed):
    """Rebuild one PultreeDataset sample at an arbitrary node_pad.

    Labels are skipped entirely: IPGN_inference only reads x, edge_index,
    points and vol_size, so none of the edge/node class logic is needed here.
    """
    volume = np.load(npz_path)                         # lazy npz handle
    volume = volume[volume.files[0]]                   # repo only ever reads the first array
    g = nx.read_graphml(gml_path)                      # igraph writes a MultiGraph

    edges = []                                         # both directions, as the repo's _adj walk produces
    for start in g._adj:                               # iterate the adjacency exactly like process()
        for endpoint in g._adj[start]:
            if start == endpoint:                      # repo skips self-loops
                continue
            edges.append([int(start[1:]), int(endpoint[1:])])   # 'n42' -> 42

    nodes = []                                         # node coords in volume axis order
    for node in g._node:                               # iteration order defines the node index
        z = int(float(g._node[node]['Z']))             # axis 0
        y = int(float(g._node[node]['Y']))             # axis 1
        x = int(float(g._node[node]['X']))             # axis 2
        nodes.append([z, y, x])

    nodes = torch.tensor(nodes, dtype=torch.float)     # (N, 3)
    if nodes.shape[0] > node_pad:                      # guard: silent truncation would fake a result
        raise ValueError(f"{nodes.shape[0]} nodes exceeds node_pad={node_pad}")

    nodes_padded = torch.ones((node_pad, 3)) * (-10)   # -10 is the repo's padding sentinel
    nodes_padded[:nodes.shape[0]] = nodes              # real nodes first, padding trails

    points = np.transpose(np.nonzero(volume))          # every foreground voxel, (M, 3)
    rng = np.random.RandomState(point_seed)            # fixed stream so runs are comparable
    points = points[rng.permutation(points.shape[0])]  # repo shuffles before slicing

    data = Data(
        x=nodes_padded,                                                    # (node_pad, 3)
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(), # (2, E)
        points=points[:TOTAL_SAMPLE_POINTS],                               # point-cloud branch
        implicit_points=points[TOTAL_SAMPLE_POINTS:TOTAL_SAMPLE_POINTS + IMPLICIT_POINTS],
        vol_size=volume.shape,                                             # for the output grid
    )
    return aug.minmax_norm_g(data), volume             # repo's transform; also return raw volume


def predict(data, volume, model, device, fps_seed):
    """Label every foreground voxel, mirroring IPGN_inference()."""
    foreground = np.transpose(np.array(np.where(volume > 0)), (1, 0))   # (M, 3) voxel coords
    maxs = np.amax(foreground, axis=0)                 # per-axis bbox max
    mins = np.amin(foreground, axis=0)                 # per-axis bbox min
    query = (foreground - mins) / (maxs - mins)        # to the unit cube, as the repo does
    query = torch.stack([torch.from_numpy(query)], dim=0)               # add the batch dim

    pts = torch.from_numpy(data.points).float().unsqueeze(0).to(device) # (1, 6000, 3)
    nodes = data.x.view(1, -1, 3).float().to(device)                    # (1, node_pad, 3)
    edge_index = data.edge_index.to(device)                             # (2, E)

    torch.manual_seed(fps_seed)                        # farthest-point sampling starts from randint
    with torch.no_grad():                              # no gradients needed for inference
        labels = model(query, pts, nodes, edge_index, full_voxel=True) + 1   # shift 0-18 -> 1-19
    return labels                                      # numpy array, one label per foreground voxel


def load_model(node_pad, device, checkpoint):
    """Build IPGN from the repo specs and load the pretrained weights."""
    with open(os.path.join(BASE_DIR, 'specs', 'network_specs.json')) as f:
        network_specs = json.load(f)                   # architecture config

    model = pg_model.IPGN(network_specs, num_class=19, max_node=node_pad, device=device).to(device)
    if os.path.exists(checkpoint):                     # normal path
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state)                   # pretrained IPGN weights
    else:
        print(f"WARNING: {checkpoint} not found - running with RANDOM weights (plumbing test only)")
    model.point_graph_network.implicit_inference_mode()   # stage 3, no grads
    model.eval()                                       # disable any train-time behaviour
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz', required=True, help='path to one .npz volume')
    parser.add_argument('--graphml', default=None, help='defaults to the repo path rule')
    parser.add_argument('--big-pad', type=int, default=850, help='pad to compare against 519')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', default=os.path.join(BASE_DIR, 'ckp', 'airway',
                                                             'Implicit_Point_Graph_Network'))
    args = parser.parse_args()

    gml = args.graphml                                 # explicit override wins
    if gml is None:                                    # otherwise apply the repo's substitution
        gml = args.npz.replace('data_npz', 'data_graph').replace('.npz', '.graphml')

    device = torch.device(args.device)                 # cuda:0 or cpu
    print(f"device: {device}\ncase: {os.path.basename(args.npz)}")

    runs = {                                           # (node_pad, point_seed) per run
        'A pad=519 seed=0': (850, 0),
        f'B pad={args.big_pad} seed=0': (args.big_pad, 0),
        'C pad=519 seed=1': (850, 1),
    }

    predictions = {}                                   # run label -> per-voxel class array
    for name, (pad, seed) in runs.items():
        data, volume = build_sample(args.npz, gml, pad, seed)   # rebuild at this padding
        model = load_model(pad, device, args.checkpoint)        # max_node affects no weights
        predictions[name] = predict(data, volume, model, device, fps_seed=0)   # same FPS seed throughout
        print(f"{name}: {len(predictions[name])} voxels labelled")

    names = list(runs)                                 # A, B, C in insertion order
    padding_effect = (predictions[names[0]] != predictions[names[1]]).mean()   # A vs B
    noise_floor = (predictions[names[0]] != predictions[names[2]]).mean()      # A vs C

    print(f"\npadding effect (A vs B): {padding_effect:.3%}")
    print(f"noise floor  (A vs C): {noise_floor:.3%}")
    if padding_effect <= max(noise_floor, 0.005):      # within sampling noise, or under 0.5%
        print("-> padding is not a meaningful factor; a global pad increase is safe")
    else:
        print("-> padding shifts predictions beyond noise; keep the per-sample pad")


if __name__ == '__main__':
    main()
