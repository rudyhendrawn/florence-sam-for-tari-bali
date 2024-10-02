import torch

class DeviceManager:
	def __init__(self):
		# Selectthe device for computation
		if torch.cuda.is_available():
			self.device = torch.device("cuda")
		elif torch.backends.mps.is_available():
			self.device = torch.device("mps")
		else:
			self.device = torch.device("cpu")
		print(f"Using device: {self.device}")

		# Set up device-specific configuration
		if self.device.type == "cuda":
			# Use bfloat16 for the entire code
			self.autocast = torch.autocast("cuda", dtype=torch.bfloat16)
			# Turn on tf32 for Ampere GPUs
			if torch.cuda.get_device_properties(0).major >= 8:
				torch.backends.cuda.matmul.allow_tf32 = True
				torch.backends.cudnn.allow_tf32 = True
		elif self.device.type == "mps":
			self.autocast = torch.autocast("cpu")
			print(
				"\nSupport for MPS devices is preliminary. SAM2 is trained with CUDA and might "
				"give numerically different outputs and sometimes degraded performance on MPS."
			)
		else:
			self.autocast = torch.autocast("cpu")

	def get_device(self):
		return self.device

	def get_autocast(self):
		return self.autocast

	def __enter__(self):
		self.autocast.__enter__()
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.autocast.__exit__(exc_type, exc_value, traceback)
