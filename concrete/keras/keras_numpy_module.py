
""" A keras to numpy module. """ 

import tensorflow.keras as keras 
import tensorflow.keras.layers as layers 
import tensorflow.keras.activations as activations 


class KerasNumpyModule: 
	""" General interface to transform a tensorflow.keras module to numpy module.""" 

	IMPLEMENTED_MODULES = {layers.Dense, layers.Flatten} 


	def __init__(self, keras_model): 
		"""Initialize our Keras numpy module. 

		Current constraint: # see NumpyModule 

		Args: 
			keras_model : A fully trained, keras model along with its parameters 
		""" 
		self.keras_model = keras_model 
		self.check_compatibility() 
		self.convert_to_numpy() 

	def check_compatibility(self): 
		"""Check the compatibility of all layers in the keras model.""" 

		for layer in self.keras_model.layers : 
			if (layer_type := type(layer)) not in self.IMPLEMENTED_MODULES:
				raise ValueError(
					f"The following module is currently not implemented: {layer_type.__name__}. "
					f"Please stick to the available keras modules: "
					f"{', '.join(sorted(module.__name__ for module in self.IMPLEMENTED_MODULES))}."
				)
		return True 


	def convert_to_numpy(self): 
		""" Transform all parameters from keras tensors to numpy arrays. """ 

		self.numpy_module_dict = {} 

		for layer in self.keras_model.layers: 
			name = layer.name  
			layer_type = type(layer) 
			if hasattr(layer, "weights"): 
				if layer.weights != []:  
					weights = layer.weights[0] 
					bias = layer.weights[1] 

				else: 
					weights = None 
					bias = None 
			if hasattr(layer, "activation"): 
				activation = layer.activation.__name__  
			else: 
				activation = None 

			self.numpy_module_dict[name] = {"type": layer_type,
											"weights":weights, 
											"bias": bias, 
											"activation": activation}  


	def __call__(self, x: numpy.ndarray): 
		""" Return the function to be compiled.""" 
		return self.forward(x) 

	def forward(self, x: numpy.ndarray) -> numpy.ndarray:  
		"""Apply a forward pass with numpy function only. 

		Args: 
			x (numpy.array): Input to be processed in the forward pass. 

		Returns: 
			x (numpy.array): Processed input. 

		""" 

		for layer_name in self.numpy_module_dict: 
			if self.numpy_module_dict[layer_name]["type"] == layers.Dense: 

				x = (x @ self.numpy_module_dict[layer_name]["weights"] + self.numpy_module_dict[layer_name]["bias"]) 

				if self.numpy_module_dict[layer_name]["activation"] == "relu": 
					x = numpy.maximum(0, x) 

				if self.numpy_module_dict[layer_name]["activation"] == "relu6": 
					x = numpy.minimum(numpy.maximum(0,x),6) 

				elif self.numpy_module_dict[layer_name]["activation"] == "sigmoid": 
					x = 1 / (1 + numpy.exp(-x))

				elif self.numpy_module_dict[layer_name]["activation"] == "softmax": 
					x = numpy.exp(x) / numpy.sum(numpy.exp(x)) 

			elif self.numpy_module_dict[layer_name]["type"] == layers.Flatten: 
				x = x.flatten() 
				x = x.reshape(1, x.shape[0])

		return x 