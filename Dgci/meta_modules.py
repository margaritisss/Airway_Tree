import torch                                                                                     # Core tensor library; used here for prod/tensor and no_grad.
from torch import nn                                                                             # Layers, containers (ModuleList) and the init functions.
from collections import OrderedDict                                                              # Keeps predicted parameters in insertion order so names line up with the hypo-module's layers.
import DCI_Modules                                                                               # The repo's SIREN-style building blocks; FCBlock is taken from here.

class HyperNetwork(nn.Module):                                                                       # A network that OUTPUTS the weights of another network instead of outputting data.
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):  # In DGCI this is built 3x: for deform_encoder, deform_decoder and sdf_net.

        '''Args:   hyper_in_features:     In features of hypernetwork
                   hyper_hidden_layers:   Number of hidden layers in hypernetwork
                   hyper_hidden_features: Number of hidden units in hypernetwork
                   hypo_module: MetaModule. The module whose parameters are predicted.'''        # Original docstring. "hypo" = the target/predicted network; here latent_dim=128 comes in as hyper_in_features.
        super().__init__()                                                                       # Registers this as an nn.Module so the sub-networks below are tracked.

        hypo_parameters   = hypo_module.meta_named_parameters()                                  # GENERATOR of (name, tensor) over the hypo-module's replaceable weights/biases; MetaModule allows these to be swapped at forward time. One-shot, consumed by the loop below.
        self.names        = []                                                                   # Parameter names, e.g. 'net.0.0.weight' — the keys the hypo-module expects back.
        self.nets         = nn.ModuleList()                                                      # One small MLP per parameter tensor; ModuleList so they register as submodules and get optimized.
        self.param_shapes = []                                                                   # The original shape of each parameter, needed to reshape the flat MLP output.

        for name, param in hypo_parameters:                                                      # Walk every weight and bias of the target network, one at a time.
            self.names.append(name)                                                              # Remember the key.
            self.param_shapes.append(param.size())                                               # Remember the shape, e.g. torch.Size([128, 4]).
            #the MLP is created here
            hn = DCI_Modules.FCBlock(in_features      = hyper_in_features,                            # A plain ReLU MLP taking the 128-D latent code as input.
                                     out_features     = int(torch.prod(torch.tensor(param.size()))),  # Output width = TOTAL element count of that parameter (128*4 = 512, etc.). This is why hypernetworks are so large.
                                     num_hidden_layers= hyper_hidden_layers, 
                                     hidden_features  = hyper_hidden_features,                          # Depth/width of the hypernetwork itself (1 layer x 256 units for the deform nets, 3 layers for the SDF net).
                                     outermost_linear = True,
                                     nonlinearity     = 'relu')                                    # Linear output (no activation squashing the predicted weights); ReLU inside.
            self.nets.append(hn)                                                                     # Register this predictor.

            if 'weight' in name:                                                                 # Different init depending on whether this MLP predicts a weight matrix...
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))    # ...applied to the LAST layer of the just-added hypernetwork. param.size()[-1] is the fan-in of the target layer, passed so the predicted weights start at a sane scale.
            elif 'bias' in name:                                                                 # ...or a bias vector.
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))                        # Same idea, but the scale is derived from the hypernetwork's own fan-in.

    def forward(self, z):                                                                        # z is the latent code(s); in DGCI it is (B+1, 128) — the template code stacked on top of the batch's subject codes.
        '''
        Args: z:  Embedding. Input to hypernetwork. Could be output of "Autodecoder"
        Returns:  params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''                                                                                      # Original docstring: the output is a full state-dict-like set of weights.
        params = OrderedDict()                                                                   # Will map parameter name → predicted tensor.
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):             # remember that in __init__ we made three lists that line up: the name of each weight ("layer 1's weight"), the little MLP that predicts it, and its shape. This loop walks all three together, one parameter at a time.
            batch_param_shape = (-1,) + param_shape                                              # the shape we want the answer in, with an extra slot at the front for "how many subjects". If the real shape is (128, 4), this becomes (-1, 128, 4). The -1 means "figure this number out for me."
            params[name] = net(z).reshape(batch_param_shape)                                     # Runs the little MLP on the latent code(s) and folds the flat output back into the parameter's real shape. Each subject in the batch gets ITS OWN weight matrix.
        return params                                                                            # Fed straight into e.g. self.sdf_net({'coords': coords}, params=sdf_hypo_params).

# The two functions below only run once, at the very beginning, before any training. 
# They set the starting values of the last layer of each little MLP.

def hyper_weight_init(m, in_features_main_net):                                                  # Init for a hypernetwork output layer that predicts a WEIGHT matrix.
    if hasattr(m, 'weight'):                                                                     # .apply() visits every submodule, including activations that have no weight — hence the guard.
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')             # Standard He init for the ReLU hypernetwork.
        m.weight.data = m.weight.data / 1.e2                                                     # Then shrinks it 100x. KEY TRICK: at step 0 the latent-dependent term is nearly zero, so the predicted weights are dominated by the bias below — every subject starts from almost the same target network, and differentiation is learned gradually. Without this, training diverges.

    if hasattr(m, 'bias'):                                                                       # The bias of this output layer IS the default weight matrix of the target layer.
        with torch.no_grad():                                                                    # Init must not be recorded in the autograd graph.
            m.bias.uniform_(-1 / in_features_main_net, 1 / in_features_main_net)                 # Scales by the TARGET layer's fan-in, mirroring PyTorch's default Linear init so the predicted network behaves like a normally-initialized MLP.

def hyper_bias_init(m):                                                                          # Init for a hypernetwork output layer that predicts a BIAS vector.
    if hasattr(m, 'weight'):                                                                     # Same guard as above.
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')             # He init...
        m.weight.data = m.weight.data / 1.e2                                                     # ...also damped 100x, for the same "start everyone identical" reason.

    if hasattr(m, 'bias'):                                                                       # This bias becomes the target layer's default bias.
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)                              # Reads the fan-in of the HYPERNETWORK's own last layer (not the target net) — the one asymmetry with hyper_weight_init, since a bias vector has no meaningful fan-in of its own.
        with torch.no_grad():                                                                    # No autograd tracking.
            m.bias.uniform_(-1 / fan_in, 1 / fan_in)                                             # Small uniform init so predicted biases start near zero.