import torch
import device_manager
from transformers import AutoProcessor, AutoModelForCausalLM, BertTokenizer, BertModel


class LanguageModel:
	def __init__(self, device_manager):
		# Load a pre-trained LLM tokenizer and model
		self.device = device_manager.get_device()
		self.model = AutoModelForCausalLM.from_pretrained(
			"microsoft/Florence-2-base", 
			torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
			trust_remote_code=True,
			).to(self.device)
		self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
		self.prompt = "<Caption>" # Task prompt for the image captioning
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.text_model = BertModel.from_pretrained("bert-base-uncased").to(self.device)

	def generate_descriptions(self, features):
		# images: list of PIL Images
		inputs = self.processor(text=self.prompt, images=images, return_tensors="pt").to(self.device)
		generated_ids = self.model.generate(
			input_ids=inputs9["input_ids"],
			pixel_values=inputs["pixel_values"],
			max_new_tokens=50,
			do_sample=False,
			num_beams=3,
		)

		generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)

		# Post-process the generated texts to extract captions
		captions = [
			self.processor.post_process_generation(text, task="<Caption>")
			for text in generated_texts
		]

		return captions

	def extract_text_features(self, captions):
		# Extract textual features from captions
		inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
		outputs = self.model(**inputs)
		text_features = outputs.pooler_output	# Shape: (batch_size, hidden_size)
		
		return text_features