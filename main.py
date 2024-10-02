
import os
import torch
import pandas as pd
import video_classification_model

from torchvision import transforms
from video_dataset import VideoDataset
from object_segmentation_model import ObjectSegmentationModel

def get_full_path(dataframe):
	paths = dataframe['path'].apply(lambda x: os.path.join(video_base_dir, x)).tolist()
	labels = df['classname'].map(class_to_idx).tolist()
	return paths, labels

def main():
	# Load data
	train_df = pd.read_csv('train.csv')
	val_df = pd.read_csv('val.csv')
	test_df = pd.read_csv('test.csv')

	# Base directory where videos are stored
	video_base_dir = 'videos/'

	# Map class names to integer labels
	class_names = sorted(train_df['classname'].unique())
	class_to_idx = {classname: idx for idx, classname in enumerate(class_names)}
	idx_to_class = {idx: classname for classname, idx in class_to_idx.items()}

	train_video_paths, train_labels = get_full_path(train_df)
	val_video_paths, val_labels = get_full_path(val_df)
	test_video_paths, test_labels = get_full_path(test_df)

	# Initialize segmentation model
	segmentation_model = ObjectSegmentationModel(checkpoint_path='sam_vit_h_14.pth', model_type='vit_h')

	# Define image transformations
	image_transforms = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	# Prepare datasets
	train_dataset = VideoDataset(train_video_paths, train_labels, transform=image_transforms, segmentation_model=segmentation_model)
	val_dataset = VideoDataset(val_video_paths, val_labels, transform=image_transforms, segmentation_model=segmentation_model)
	test_dataset = VideoDataset(test_video_paths, test_labels, transform=image_transforms, segmentation_model=segmentation_model)

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