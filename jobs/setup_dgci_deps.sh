#!/bin/bash
# Run ONCE on a login node, with the venv active. Not a SLURM job.
#
#   source /home/ids/gmargari-24/airway_project/new_3env/bin/activate
#   bash setup_dgci_deps.sh
#
# Installs the 4 easy missing packages and builds a minimal torchmeta shim.
# Does NOT modify any DGCI source file.
set -e

SHIM_ROOT=/home/ids/gmargari-24/airway_project/shims

echo "=== 1/2  pip packages ==="
pip install configargparse plyfile einops tensorboard

echo
echo "=== 2/2  torchmeta shim -> $SHIM_ROOT ==="
# Why a shim: real torchmeta 1.8.0 pins torch<1.10 (unusable on an A100), and
# its __init__ imports torchmeta.datasets, which calls torchvision private
# functions that were removed. Separately, DCI_Modules.py imports
# torchmeta.modules.utils.get_subdict, which torchmeta DELETED in v1.5 -- so no
# installable version satisfies this code. These four files are the only pieces
# DGCI touches, taken from torchmeta 1.4.6.
mkdir -p "$SHIM_ROOT/torchmeta/modules"
cd "$SHIM_ROOT/torchmeta"

cat > __init__.py <<'EOF'
"""Minimal stand-in for torchmeta: only what DGCI's DCI_Modules.py imports."""
EOF

cat > modules/module.py <<'EOF'
import re
import warnings
from collections import OrderedDict

import torch.nn as nn


class MetaModule(nn.Module):
    """Base class for meta-learning modules; forward() accepts a `params` dict."""

    def __init__(self):
        super(MetaModule, self).__init__()
        self._children_modules_parameters_cache = dict()

    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        if params is None:
            return None

        all_names = tuple(params.keys())
        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names
            else:
                key_escape = re.escape(key)
                key_re = re.compile(r'^{0}\.(.+)'.format(key_escape))
                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r'\1', k) for k in all_names
                    if key_re.match(k) is not None]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn('Module `{0}` has no parameter corresponding to the '
                          'submodule named `{1}` in the dictionary `params`.'
                          .format(self.__class__.__name__, key), stacklevel=2)
            return None

        return OrderedDict([(name, params['{0}.{1}'.format(key, name)])
                            for name in names])
EOF

cat > modules/container.py <<'EOF'
import torch.nn as nn

from torchmeta.modules.module import MetaModule


class MetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=self.get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError('The module must be either a torch module '
                                '(inheriting from `nn.Module`), or a '
                                '`MetaModule`. Got type: `{0}`'.format(type(module)))
        return input
EOF

cat > modules/utils.py <<'EOF'
import re
from collections import OrderedDict


def get_subdict(dictionary, key=None):
    """Module-level get_subdict, verbatim from torchmeta 1.4.6.

    Removed upstream in torchmeta 1.5, but DCI_Modules.py imports it by name.
    The deprecation warning is dropped: this runs several times per forward
    pass and would flood the log.
    """
    if dictionary is None:
        return None

    if (key is None) or (key == ''):
        return dictionary

    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    # Compatibility with DataParallel, which prefixes names with 'module.'
    if not any(filter(key_re.match, dictionary.keys())):
        key_re = re.compile(r'^module\.{0}\.(.+)'.format(re.escape(key)))

    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
                       in dictionary.items() if key_re.match(k) is not None)
EOF

cat > modules/__init__.py <<'EOF'
from torchmeta.modules.container import MetaSequential
from torchmeta.modules.module import MetaModule

__all__ = ['MetaModule', 'MetaSequential']
EOF

echo
echo "=== verifying ==="
PYTHONPATH="$SHIM_ROOT:$PYTHONPATH" python - <<'EOF'
import importlib
mods = ['torch','scipy','yaml','configargparse','tqdm','numpy',
        'torchmeta.modules','einops.layers.torch','skimage.measure',
        'plyfile','torch.utils.tensorboard']
bad = []
for m in mods:
    try:
        importlib.import_module(m); print('  OK      ', m)
    except Exception as e:
        print('  MISSING ', m, '->', type(e).__name__, e); bad.append(m)
from torchmeta.modules import MetaModule, MetaSequential
from torchmeta.modules.utils import get_subdict
print('\n  shim exports MetaModule / MetaSequential / get_subdict: OK')
raise SystemExit(1 if bad else 0)
EOF

echo
echo "DONE. Add this line to your sbatch, after the 'cd \$DGCI_DIR':"
echo "    export PYTHONPATH=$SHIM_ROOT:\$PYTHONPATH"
