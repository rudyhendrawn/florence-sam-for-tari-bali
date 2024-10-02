import torch
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from torchvision import transforms
from language_model import LanguageModel
from feature_extractor import FeatureExtractor
from multimodal_feature_fusion import MultimodalFeatureFusion

class VideoClassificationModel(pl.LightningModule):
	def __init__(self, num_classes):
		super(VideoClassificationModel, self).__init__()
		self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.feature_extractor = FeatureExtractor(device=self.device_)
		self.language_model = LanguageModel(device=self.device_)
		visual_dim = 1024 # Based on Florence-2's encoder output size
		textual_dim = 768 # Hidden size of the textual features, for BERT-base
		fused_dim = 512
		self.fusion = MultimodalFeatureFusion(visual_dim=visual_dim, textual_dim=textual_dim, fused_dim=fused_dim)
		self.classifier = nn.Linear(fused_dim, num_classes)
		self.loss_fn = nn.CrossEntropyLoss()

	def forward(self, frames):
		# Assume frames are of shape (batch_size, num_frames, channels, height, width)
		batch_size, num_frames, C, H, W = frames.size()
		frames = frames.view(-1, C, H, W)	# Flatten frames for CNN
		images = [transforms.ToPILImage()(frame) for frame in frames]

		# Extract visual features
		visual_features = self.feature_extractor(images)	# (batch_size * num_frames, 2048)
		visual_features = visual_features.view(batch_size, num_frames, -1).mean(dim=1)	# Average pooling

		# Generate captions (placeholder)
		captions = []
		for i in range(0, len(images), 4): 	# Adjust batch size based on GPU memory
			batch_images = images[i:i+4]
			batch_captions = self.language_model.generate_descriptions(batch_images)
			captions.extend(batch_captions)

		# Extract textual features
		textual_features = self.language_model.extract_text_features(captions)
		textual_features = textual_features.view(batch_size, num_frames, -1).mean(dim=1)	# Average pooling

		# Fuse visual and textual features
		fused_features = self.fusion(visual_features, textual_features)

		# Classify the fused features
		logits = self.classifier(fused_features)
		return logits

	def training_step(self, batch, batch_idx):
		frames, labels = batch
		logist = self
		loss = self.loss_fn(logits, labels)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, batch, batch_idx):
		frames, labels = batch
		logits = self(frames)
		loss = self.loss_fn(logits, labels)
		preds = torch.argmax(logits, dim=1)
		acc = (preds == labels).float().mean()
		self.log('val_loss', loss)
		self.log('val_acc', acc)

	def test_step(self, batch, batch_idx):
		frames, labels = batch
		logist = self(frames)
		loss = self.loss_fn(logits, labels)
		preds = torch.argmax(logits, dim=1)
		acc = (preds == labels).float().mean()
		self.log('test_loss', loss)
		self.log('test_acc', acc)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
		return optimizer