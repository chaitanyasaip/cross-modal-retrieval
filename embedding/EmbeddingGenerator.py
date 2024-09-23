from transformers import AutoModel, AutoTokenizer, ClapModel, ClapProcessor
import torch

class EmbeddingGenerator:
    def __init__(self, model_name='laion/clap-htsat-fused'):
        
        #self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load CLAP model and processor
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)

    def embed_text(self, text):
        """
        Generate embedding for text input.
        """
        inputs = self.processor(text=[text], return_tensors='pt').to(self.device) # Convert text to tokens and add batch dimension
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs)
        return embedding.cpu()

    def embed_audio(self, audio):
        """
        Generate embedding for audio input.
        """
        # Process audio input
        inputs = self.processor(audios=[audio], sampling_rate=48000, return_tensors='pt').to(self.device)
        with torch.no_grad():
            embedding = self.model.get_audio_features(**inputs)
        return embedding.cpu()

    def embed_audio_batch(self, audio_list):
        """
        Generate embeddings for a batch of audio inputs.
        """
        # Process audio inputs
        inputs = self.processor(audios=audio_list, sampling_rate=48000, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_audio_features(**inputs)
        return embeddings.cpu()