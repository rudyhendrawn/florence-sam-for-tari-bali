import cv2
import torch
import numpy as np
import device_manager

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
	def __init__(self, video_paths, labels, transform=None, segmentation_model=None, device_manager=None):
		self.video_paths = video_paths
		self.labels = labels
		self.transform = transform
		self.segmentation_model = segmentation_model
		self.device = device_manager.get_device()
	
	def __len__(self):
		return len(self.video_paths)

	def __getitem__(self, idx):
		video_path = self.video_paths[idx]
		label = self.labels[idx]

		frames = self.extract_frames(video_path)

		if self.segmentation_model:
			frames = self.segmentation_model.segment_objects(frames)

		if self.transform:
			frames = [self.transform(frame) for frame in frames]
		
		frames_tensor = torch.stack(frames)
		label = torch.tensor(label).to(self.device)

		return frames_tensor, label

	def extract_frames(self, video_path):
		# Frame extraction
		cap = cv2.VideoCapture(video_path)
		frames = []
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		if frame_count <= 0:
			raise ValueError(f"Video {video_path} has no frames")
		frame_idxs = np.linspace(0, frame_count - 1, num=16, dtype=int)

		for idx in frame_idxs:
			cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
			ret, frame = cap.read()
			
			if not ret:
				break

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frames.append(frame)
		cap.release()

		if len(frames) < 16:
			# Duplicate last frame to have 16 frames
			while len(frames) < 16:
				frames.append(frames[-1])

		return frames