
"""A library implementing network modulation
techniques (injecting embeddings, etc)


Authors
 * Artem Ploujnikov
"""

from torch import nn
from speechbrain.utils.data_utils import unsqueeze_match_feature_dim

class Modulation(nn.Module):
    """A common interface for modulation methods"""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input, mod_input=None):
        return NotImplementedError()


class InputModulation(Modulation):
    """Simple residual modulation (passes the modulation input through a simple
    linear layer with activation and adds it to the model
    input)"""

    def __init__(self, module, activation, input_size, mod_size=None, feature_dim=-1, **kwargs):
        super().__init__(module)
        self.input_size = input_size
        self.mod_size = mod_size
        self.proj = nn.Linear(mod_size, self.input_size)
        self.feature_dim = feature_dim
        if activation is None:
            activation = nn.SiLU()
        self.activation = activation

        
    def forward(self, input, mod_input=None):
        """Computes the modulation
        
        Arguments
        ---------
        input: torch.Tensor
            the model input
        mod_input: torch.tensor
            the input that modulates the network
        """
        if mod_input is None:
            mod_input = input        
        mod_proj = self.proj(mod_input)
        mod_proj = self.activation(mod_proj)
        mod_proj = unsqueeze_match_feature_dim(
            mod_proj, input, self.feature_dim
        )
        modulated_input = input + mod_proj
        return self.module(modulated_input)


class OutputModulation(Modulation):
    """A simple output modulation that passes the modulation input
    through a linear layer with an activation and then adds it to the
    model output
    
    Arguments
    ---------
    module: callable
        the module to be wrapped
    output_size: int
        the size of the module output
    activation: callable
        the activation module to use
    mod_size: int
        the size of the modulation tensor
    feature_dim: int
        the feature dimension
    """
    def __init__(self, module, output_size, activation=None, mod_size=None, feature_dim=-1, **kwargs):
        super().__init__(module)
        self.output_size = output_size
        self.mod_size = mod_size
        self.proj = nn.Linear(mod_size, self.output_size)
        self.feature_dim = feature_dim
        if activation is None:
            activation = nn.SiLU()
        self.activation = activation

    def forward(self, input, mod_input=None):
        """Applies the modulation
        
        Arguments
        ---------
        input: torch.Tensor
            the model input
        mod_input: torch.tensor
            the modulation input

        Returns
        -------
        output: torch.Tensor
            the module output
        """
        mod_proj = self.proj(mod_input)
        mod_proj = self.activation(mod_proj)
        output = self.module(input)
        mod_proj = unsqueeze_match_feature_dim(
            mod_proj, output, self.feature_dim
        )
        modulated_output = output + mod_proj
        return modulated_output


class NullModulation(Modulation):
    """A no-op modulation implementation that completely ignores
    the modulation tensor
    
    Arguments
    ---------
    module: callable
        the module to be wrapped
    """
    def __init__(self, module, **kwargs):
        super().__init__(module)

    def forward(self, input, mod_input=None):
        """Applies the modulation
        
        Arguments
        ---------
        input: torch.Tensor
            the model input
        mod_input: torch.tensor
            the modulation input

        Returns
        -------
        output: torch.Tensor
            the module output
        """        
        return self.module(input)


class FiLM(Modulation):
    """Feature-wise Linear Modulation (FiLM)
    https://arxiv.org/pdf/1709.07871.pdf

    Arguments
    ---------
    module: callable
        the module being wrapped

    input_size: int
        the input dimension
    output_size: int
        the output dimension
    """
    def __init__(self, module, input_size, output_size, mod_size=None, feature_dim=-1, **kwargs):
        super().__init__(module)
        self.input_size = input_size
        self.output_size = output_size
        if mod_size is None:
            mod_size = input_size
        self.mod_size = mod_size

        self.gamma = nn.Linear(self.mod_size, self.input_size)
        self.beta = nn.Linear(self.mod_size, self.output_size)
        self.feature_dim = feature_dim

    def forward(self, input, mod_input=None):
        if mod_input is None:
            mod_input = input
        
        x = input
        gamma_out = self.gamma(mod_input)
        gamma_out = unsqueeze_match_feature_dim(gamma_out, x, self.feature_dim)
        x = gamma_out * x
        x = self.module(x)
        beta_out = self.beta(mod_input)
        beta_out = unsqueeze_match_feature_dim(beta_out, x, self.feature_dim)
        x += beta_out
        return x


MODULATIONS = {
    "input": InputModulation,
    "output": OutputModulation,
    "film": FiLM,
    None: NullModulation
}

def modulate(module, modulation, *args, **kwargs):
    """Wrap the module to use the specified modulation
    
    Arguments
    ---------
    module: torch.nn.Module
        the module to wrap
    mod_type: str
        the modulation type
    args: list
        modulation arguments
    kwargs: dict
        keyword arguments
    """
    if modulation is None or isinstance(modulation, str):
        modulation = MODULATIONS[modulation]
    elif not isinstance(modulation, callable):
        raise ValueError("modulation must be a string or a module")
    return modulation(module, *args, **kwargs)
    







