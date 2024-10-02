import torch
import torch.nn as nn
import device_manager
from transformers import AutoProcessor, AutoModelForCausalLM

class FeatureExtractor(nn.Module):
	def __init__(self, device_manager):
		super(FeatureExtractor, self).__init__()
		self.device = device_manager.get_device()
		self.model = AutoModelForCausalLM.from_pretrained(
			"microsoft/Florence-2-base", 
			torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
			trust_remote_code=True,
			).to(self.device)
		self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

	def forward(self, x):
		# Images: list of images as Numpy arrays or PIL images
		inputs = self.processor(images=images, return_tensors="pt").to(self.device)
		# Use the encoder to get visual features
		outputs = self.model.get_encoder()(pixel_values=inputs["pixel_values"])
		# Extract the last hidden state
		visual_features = outputs.last_hidden_state.mean(dim=1) 	# Shape: (batch_size, hidden_size)
		return visual_features