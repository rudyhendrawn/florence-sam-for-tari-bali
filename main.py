
import os
import hydra
import torch
import pandas as pd
import video_classification_model

from omegaconf import DictConfig
from torchvision import transforms
from video_dataset import VideoDataset
from device_manager import DeviceManager
from hydra import initialize_config_dir, compose
from object_segmentation_model import ObjectSegmentationModel
from hydra.core.global_hydra import GlobalHydra

def get_full_path(dataframe, video_base_dir, class_to_idx):
	paths = dataframe['path'].apply(lambda x: os.path.join(video_base_dir, x)).tolist()
	labels = dataframe['classname'].map(class_to_idx).tolist()
	return paths, labels

@hydra.main(config_path='configs', config_name='sam2.1_hiera_s.yaml', version_base=None)
def main(cfg: DictConfig) -> None:
	GlobalHydra.instance().clear()

	# Base directory where videos are stored
	video_base_dir = '/Users/rudyhendrawan/miniforge3/datasets/small-Dasar-Gerakan-Tari-Bali-All-Women'

	# Initialize device manager
	with DeviceManager() as device_manager:
		device = device_manager.get_device()
	
		# Load data
		train_df = pd.read_csv(os.path.join(video_base_dir, 'train_very_small.csv'))
		val_df = pd.read_csv(os.path.join(video_base_dir, 'val_very_small.csv'))
		test_df = pd.read_csv(os.path.join(video_base_dir, 'test_very_small.csv'))

		# Map class names to integer labels
		class_names = sorted(train_df['classname'].unique())
		class_to_idx = {classname: idx for idx, classname in enumerate(class_names)}
		idx_to_class = {idx: classname for classname, idx in class_to_idx.items()}

		train_video_paths, train_labels = get_full_path(train_df, video_base_dir, class_to_idx)
		val_video_paths, val_labels = get_full_path(val_df, video_base_dir, class_to_idx)
		test_video_paths, test_labels = get_full_path(test_df, video_base_dir, class_to_idx)

		# Path to configs and checkpoints
		# config_path = os.path.abspath('configs')
		checkpoint_path = os.path.abspath('checkpoints/sam2.1_hiera_small.pt')

		# Initialize Hydra
		# with initialize_config_dir(config_dir=config_path):
		# 	cfg = compose(config_name="sam2.1_hiera_s.yaml")

		# Initialize segmentation model
		segmentation_model = ObjectSegmentationModel(
			device_manager=device_manager, 
			model_type='video',
			config=cfg,
			checkpoint_path=checkpoint_path
		)

		# Define image transformations
		image_transforms = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		# Prepare datasets
		train_dataset = VideoDataset(
			train_video_paths, 
			train_labels, 
			transform=image_transforms, 
			segmentation_model=segmentation_model,
			device_manager=device_manager
		)
		val_dataset = VideoDataset(
			val_video_paths, 
			val_labels, 
			transform=image_transforms, 
			segmentation_model=segmentation_model,
			device_manager=device_manager
		)
		test_dataset = VideoDataset(
			test_video_paths, 
			test_labels, 
			transform=image_transforms, 
			segmentation_model=segmentation_model,
			device_manager=device_manager
		)

		# Data loaders
		train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
		val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
		test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

		# Initialize model
		num_classes = len(class_names)
		model = VideoClassificationModel(num_classes=num_classes)
		model.to(model.device_)

		# Initialize trainer
		trainer = video_classification_model.Trainer(max_epochs=3, gpus=1 if torch.cuda.is_available() else 0)

		# Train the model
		trainer.fit(model, train_loader, val_loader)

		# Test the model
		trainer.test(model, test_loader)

if __name__ == '__main__':
	main()