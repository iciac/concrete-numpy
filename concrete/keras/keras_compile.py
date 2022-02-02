"""keras compilation function.""" 

# main things to change here are just the torch.Tensor equivalent in keras 
# and to make sure the called functions actually run 

from typing import Iterable, Optional, Union 

import numpy as np 
import tensorflow as tf 
import tensorflow.keras as keras 

from ..common.compilation import CompilationArtifacts, CompilationConfiguration 
from ..quantization import PostTrainingAffineQuantization, QuantizedArray, QuantizedModule 

from . import KerasNumpyModule 

KerasDataset= Iterable[keras.Tensor] # need to change 
NPDataset = Iterable[np.ndarray] 

def convert_torch_keras_or_numpy_array_to_numpy_array(
	keras_tensor_or_numpy_array: Union[tf.Tensor, np.ndarray]
) -> np.ndarray: 
	"""Convert a keras tensor or a numpy array to a numpy array. 

    Args:
        torch_tensor_or_numpy_array (Union[torch.Tensor, numpy.ndarray]): the value that is either
            a torch tensor or a numpy array.

    Returns:
        numpy.ndarray: the value converted to a numpy array.
    """ 

    return (
    	keras_tensor_or_numpy_array 
    	if isinstance(keras_tensor_or_numpy_array, np.ndarray) 
    	else convert_keras_or_numpy_array_to_numpy_array.cpu().numpy() 
    	) 

def compile_keras_model(
	keras_model: keras_module, 
	keras_inputset: Union[KerasDataset, NPDataset], 
	compilation_configuration: Optional[CompilationConfiguration] = None, 
	compilation_artifacts: Optional[CompilationArtifacts] = None, 
	show_mlir: bool = False, 
	n_bits=7,
) -> QuantizedModule: 
	"""Take a model in keras, turn it into numpy, transform weights to integer. 

	Later, we'll compile the integer model. 

	Args: 
		keras_model (keras model): the model to quantize, 
		keras_inputset (Union[KerasDataset, NPDataset]): the inputset, can contain either torch tensors 
			or numpy.ndarray, only datasets with a signle input are supported for now. 
		function_parameters_encrypted_status (Dict[str, Union[str, EncryptedStatus]]): a dict with
			the name of the parameter and its encrypted status 
		compilation_configuration (CompilationConfiguration): Configuration object to use 
			during compilation 
		show_mlir (bool): if set, the MLIR produced by the converter and which is going 
			to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
		n_bits: the number of bits for the quantization 

	Returns: 
		QuantizedModule: The resulting comiled QuantizedModule. 
	""" 

	# Create corresponding numpy model 
	numpy_model = KerasNumpyModule(keras_model) 

	# Keras input to numpy 
	numpy_inputset_as_single_array = np.concatenate( 
		tuple(
			np.expand_dims(convert_keras_or_numpy_array_to_numpy_array(input_), 0)
			for input_ in keras_inputset 
		)
	) 

	# Quantize with post-training static method, to have a model with integer weights 
	post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_model, is_signed=True) 
	quantized_module = post_training_quant.quantize_module(numpy_inputset_as_single_array) 

	# Quantize input 
	quantized_numpy_inputset = QuantizedArray(n_bits, numpy_inputset_as_single_array) 

	quantized_module.compile(quantized_numpy_inputset, 
		compilation_configuration, 
		compilation_artifacts,
		show_mlir, 
	) 

	return quantized_module 
