import torch.nn as nn

class MultimodalFeatureFusion(nn.Module):
	def __init__(self, visual_dim, textual_dim, fused_dim):
		super(MultimodalFeatureFusion, self).__init__()
		self.visual_fc = nn.Linear(visual_dim, fused_dim)
		self.textual_fc = nn.Linear(textual_dim, fused_dim)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.1)
	
	def forward(self, visual_features, textual_features):
		v = self.relu(self.visual_fc(visual_features))
		t = self.relu(self.textual_fc(textual_features))
		fused_features = v + t	# Simple addition; can be replaced with more complex fusion
		return fused_features