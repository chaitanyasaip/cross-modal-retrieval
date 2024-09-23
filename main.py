import os
import logging
from preprocessing.preprocess_audio import preprocess_audio
from preprocessing.preprocess_text import preprocess_text
from embedding.EmbeddingGenerator import EmbeddingGenerator
from indexing.EmbeddingIndex import EmbeddingIndex

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    # Initialize embedding generator and index
    logging.info("Initializing models and index...")
    embedding_generator = EmbeddingGenerator()

    # Paths
    data_dir = 'data'
    audio_files = []
    audio_embeddings = []
    audio_tags = []
    # Process and embed audio samples
    logging.info("Processing and embedding audio samples...")
    embedding_dimension = None
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    audio, sr = preprocess_audio(file_path)
                    embedding = embedding_generator.embed_audio(audio)
                    # Print embedding shape
                    #print(f"Embedding shape: {embedding.shape}")
                    # Set embedding dimension and initialize index
                    if embedding_dimension is None:
                        embedding_dimension = embedding.shape[-1]
                        embedding_index = EmbeddingIndex(embedding_dimension=embedding_dimension)
                        logging.info(f"Embedding dimension set to: {embedding_dimension}")
                    # Verify embedding dimension
                    #assert embedding.shape[-1] == embedding_dimension, \
                    #    f"Embedding dimension mismatch: expected {embedding_dimension}, got {embedding.shape[-1]}"
                    audio_embeddings.append(embedding.squeeze(0).numpy())
                    audio_files.append(file_path)

                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")


    # Check if embeddings were generated
    if not audio_embeddings:
        raise ValueError("No audio embeddings were generated.")
    
    # Add embeddings to the index
    logging.info("Adding embeddings to the index...")
    embedding_index.add_embeddings(audio_embeddings)

    # Text query
    #text_query = "Energetic drum beats"
    #text_query = "I want to hear some bass music"
    #text_query = "I want to hear some piano music"
    text_query = "I want to hear some pop music"
    logging.info(f"Processing text query: '{text_query}'")
    preprocessed_text = preprocess_text(text_query)
    text_embedding = embedding_generator.embed_text(preprocessed_text)
    # Verify dimension
    #assert text_embedding.shape[-1] == embedding_dimension, \
    #    f"Text embedding dimension mismatch: expected {embedding_dimension}, got {text_embedding.shape[-1]}"

    # Retrieve similar audio samples
    logging.info("Retrieving similar audio samples...")
    distances, indices = embedding_index.query(text_embedding, k=5)
    retrieved_files = [audio_files[i] for i in indices[0]]
    print("\nRetrieved audio samples:")
    for idx, file in enumerate(retrieved_files):
        print(f"{idx + 1}: {file} (Distance: {distances[0][idx]:.4f})")
    
if __name__ == "__main__":
    main()