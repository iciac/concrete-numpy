""" A keras to numpy module. """ 

import numpy 
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
		self.keras_model_dict = None 
		self.check_compatibility() 
		self.convert_to_numpy() 

	def check_compatibility(self): 
		"""Check the compatibility of all layers in the keras model.""" 

		for layer in self.keras_model.layers : ### : 
			if (layer_type := type(layer)) not in self.IMPLEMENTED_MODULES:
				raise ValueError(
					f"The following module is currently not implemented: {layer_type.__name__}. "
					f"Please stick to the available keras modules: "
					f"{', '.join(sorted(module.__name__ for module in self.IMPLEMENTED_MODULES))}."
				)
		return True 

	def get_model_weights(self): 
		""" Get the Keras model weights in a dictionary. """ 

		model_dict = {layer.name: [layer.get_weights(),layer.activation.__name__, type(layer)] for layer in model.layers
		if hasattr(layer,'activation')}		
		return model_dict 

	#def get_layer_activations(self): 
	#	for layer in self.keras_model.layers: 


	def convert_to_numpy(self): 
		""" Transform all parameters from keras tensors to numpy arrays. """ 

		self.numpy_module_dict = {} 
		self.keras_model_dict = self.get_model_weights() 

		for name in self.keras_model_dict.keys(): 
			weights, bias, activation, layer_type = self.keras_model_dict[name][0][0], self.keras_model_dict[name][0][1], self.keras_model_dict[name][1], self.keras_model_dict[name][2]
			self.numpy_module_dict[name] = {"type": layer_type,
											"weights":weights, # may need transpose? 
											"bias": bias, 
											"activation": activation}  
			#self.numpy_module_dict[str(name)+"weight"] = weights.T
			#self.numpy_module_dict[str(name)+"bias"] = bias 
			#self.numpy_module_dict[str(name)+"activation"] = activation 



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

		#for name, layer in A: 
		#for layer in self.keras_model_dict: 
		#	x = ( x @ self.)

		for layer_name in self.numpy_module_dict: 
			if self.numpy_module_dict[layer_name]["type"] == layers.Dense: 
				x = (x @ self.numpy_module_dict[layer_name]["weights"] + self.numpy_module_dict[layer_name]["bias"]) 
				if self.numpy_module_dict[layer_name]["activation"] == "relu": 
					x = numpy.minimum(numpy.maximum(0, x), 6) # ReLU 6 
				elif self.numpy_module_dict[layer_name]["activation"] == "softmax": 
					x = 1 / (1 + numpy.exp(-x))

		return x 