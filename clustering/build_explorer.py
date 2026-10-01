"""Build one offline HTML explorer: method tabs, projection tabs, 2D/3D, 9 colourings.

Keep this file and explorer_template.html next to your notebooks, then:

    import build_explorer as bx
    ex = bx.Explorer("Airway latent space explorer", subjects)      # subjects in latent-row order
    ex.set_metrics(colour_table)                                    # 'subject' + the 8 metric columns
    ex.add("Agglomerative clustering", "Kernel PCA", labels,
           coords_2d=kpca_3d[:, :2], coords_3d=kpca_3d, axes="Kernel PC")
    for nn, (umap_2d, umap_3d) in runs.items():                     # one entry per hyper-parameter
        ex.add("Agglomerative clustering", "UMAP", labels, coords_2d=umap_2d, coords_3d=umap_3d,
               axes="UMAP ", parameter="n_neighbors", setting=nn)
    ex.save("latent_explorer.html")

Every method gets its own tab, every projection a tab inside it, and each plot can be shown
in 2D or 3D and coloured by the clustering or by any metric. Coordinates are stored once per
projection, so adding the 8 metric colourings costs almost nothing.
"""

import json                                                      # the viewer reads one JSON block
from pathlib import Path                                         # file paths
from html import escape                                          # safe page title

import numpy as np                                               # arrays and percentiles
from plotly.offline import get_plotlyjs                          # plotly.js source, for offline use

HERE = Path(__file__).resolve().parent                           # folder of this script
PALETTE = ["#2684f6", "#ab380b", "#11db94", "#f2b227", "#fb0000", "#008300", "#4a3aa7", "#1c0404"]
SYMBOLS_2D = ["circle"] * 8                                       # plain dots read best in 2D
SYMBOLS_3D = ["circle"] * 8                                       # plain dots read best in 3D


def _round(values, digits=5):
    """Plain python floats, rounded, so the JSON stays small."""
    return [round(float(v), digits) for v in values]


def _axis_names(axes):
    """'UMAP ' -> ['UMAP 1', 'UMAP 2', 'UMAP 3']; a list of names is passed through."""
    if isinstance(axes, str):
        return [f"{axes}{i}" for i in (1, 2, 3)]
    names = list(axes)
    return names + [f"axis {i}" for i in range(len(names) + 1, 4)]   # pad to three


class Explorer:
    """Collects methods, their clusterings and their projections."""

    def __init__(self, title="Latent space explorer", subjects=None,
                 template=HERE / "explorer_template.html"):
        self.title = title                                       # page title
        self.subjects = [str(s) for s in (subjects if subjects is not None else [])]
        self.template = Path(template)                           # the viewer page
        self.metrics = []                                        # [{name, values, vmin, vmax}]
        self.methods = []                                        # [{name, clusters, labels, projections}]

    def set_metrics(self, table, columns=None):
        """Metric values in latent-row order: a DataFrame with a 'subject' column, or a dict."""
        if hasattr(table, "columns"):                            # pandas DataFrame
            if "subject" in table.columns:                       # check the row order matches
                assert list(map(str, table["subject"])) == self.subjects, "metric rows are in a different order"
            names = columns or [c for c in table.columns if c != "subject"]
            data = {name: table[name].to_numpy(dtype=float) for name in names}
        else:                                                    # plain dict of arrays
            data = {name: np.asarray(values, dtype=float) for name, values in table.items()}
        self.metrics = []
        for name, values in data.items():
            values = np.asarray(values, dtype=float)
            assert len(values) == len(self.subjects), f"{name}: {len(values)} values for {len(self.subjects)} subjects"
            present = values[~np.isnan(values)]                  # measured values only
            if not len(present):                                 # nothing to colour by
                print(f"skipped {name}: no values")
                continue
            vmin, vmax = np.percentile(present, [1, 99])         # same crop as the notebook figures
            self.metrics.append({"name": name, "vmin": float(vmin), "vmax": float(vmax),
                                 "values": [None if np.isnan(v) else round(float(v), 6) for v in values]})
        return self

    def add(self, method, projection, labels, coords_2d=None, coords_3d=None, axes="axis ",
            parameter=None, setting=None):
        """One projection of one clustering, optionally one hyper-parameter setting of it.

        parameter: the knob's name, e.g. "n_neighbors"; setting: its value, e.g. 20.
        Calling add() again with the same method and projection but another setting adds it
        to that projection's parameter menu instead of replacing it."""
        labels = np.asarray(labels).astype(int)                  # cluster id per subject
        assert len(labels) == len(self.subjects), "labels and subjects have different lengths"
        entry = self._method(method, labels)                     # find or create the method tab
        dims = {}                                                # "2D" and "3D" coordinate blocks
        for name, coords in [("2D", coords_2d), ("3D", coords_3d)]:
            if coords is None:
                continue
            coords = np.asarray(coords, dtype=float)
            wanted = 2 if name == "2D" else 3
            assert coords.shape == (len(self.subjects), wanted), f"{method} {projection} {name}: {coords.shape}"
            dims[name] = [_round(coords[:, i]) for i in range(wanted)]   # one list per axis
        assert dims, f"{method} {projection}: pass coords_2d, coords_3d or both"
        label = "" if setting is None else str(setting)          # shown in the parameter menu
        for existing in entry["projections"]:                    # same projection again
            if existing["name"] != projection:
                continue
            existing["parameter"] = existing["parameter"] or parameter   # first name given wins
            for known in existing["settings"]:                   # same setting -> merge 2D and 3D
                if known["label"] == label:
                    known["dims"].update(dims)
                    return self
            existing["settings"].append({"label": label, "dims": dims})  # a new setting
            return self
        entry["projections"].append({"name": projection, "axes": _axis_names(axes),
                                     "parameter": parameter, "settings": [{"label": label, "dims": dims}]})
        return self

    def _method(self, name, labels):
        """Find the method tab, or create it with one colour and symbol per cluster."""
        for entry in self.methods:
            if entry["name"] == name:
                assert entry["labels"] == labels.tolist(), f"{name}: labels changed between projections"
                return entry
        ids = np.unique(labels)                                  # sorted cluster ids
        assert len(ids) <= len(PALETTE), "more clusters than colours: extend PALETTE"
        clusters = [{"name": f"cluster {c} (n={int((labels == c).sum())})", "color": PALETTE[i],
                     "symbol2d": SYMBOLS_2D[i], "symbol3d": SYMBOLS_3D[i]} for i, c in enumerate(ids)]
        remap = {c: i for i, c in enumerate(ids)}                # cluster id -> position, for the viewer
        entry = {"name": name, "clusters": clusters, "projections": [],
                 "labels": [remap[c] for c in labels.tolist()]}
        self.methods.append(entry)
        return entry

    def payload(self):
        """Everything the viewer needs, as one dict."""
        return {"title": self.title, "subjects": self.subjects,
                "metrics": self.metrics, "methods": self.methods}

    def dump(self, path):
        """Save the collected methods, to merge notebooks later."""
        Path(path).write_text(json.dumps(self.payload()), encoding="utf-8")
        return self

    def load(self, path):
        """Append the methods saved by dump() in another notebook."""
        other = json.loads(Path(path).read_text(encoding="utf-8"))
        if not self.subjects:                                    # first thing loaded wins
            self.subjects = other["subjects"]
        assert other["subjects"] == self.subjects, "the two files use different subjects or row orders"
        if not self.metrics:                                     # metrics are shared by every method
            self.metrics = other["metrics"]
        self.methods.extend(other["methods"])
        return self

    def save(self, path="latent_explorer.html"):
        """Write one self-contained HTML file: works offline, no server needed."""
        assert self.methods, "nothing to save: call add() first"
        html = self.template.read_text(encoding="utf-8")         # viewer page with placeholders
        data = json.dumps(self.payload()).replace("</", "<\\/")  # "</" would close the script tag early
        plotly_js = get_plotlyjs().replace("\ufffd", "\\uFFFD")  # raw U+FFFD -> escape, same meaning
        html = html.replace("/*__PLOTLY_JS__*/", plotly_js, 1)   # inline plotly.js (count=1: first only)
        html = html.replace("{/*__FIGURES__*/}", data, 1)        # inline the data
        html = html.replace("__TITLE__", escape(self.title))     # tab title before the data loads
        Path(path).write_text(html, encoding="utf-8")
        plots = sum(len(t["dims"]) for m in self.methods for p in m["projections"] for t in p["settings"])
        print(f"wrote {path} ({Path(path).stat().st_size / 1e6:.1f} MB, {len(self.methods)} methods, "
              f"{plots} plots x {len(self.metrics) + 1} colourings)")
        return self
