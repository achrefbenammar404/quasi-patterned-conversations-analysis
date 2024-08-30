from typing import List  , Dict , Union
import numpy as np 
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 

class ExtractEmbed : 
    def extract_customer_support_utterances(
        processed_formatted_conversations : dict
        ) ->  Dict: 
        customer_support_agent_utterances = {}
        for i , conv in enumerate(processed_formatted_conversations.values()) : 
            customer_support_agent_utterances[i] = [
                utterance["content"] for utterance in conv if utterance["role"] in["agent" , "action"]
            ]
        return customer_support_agent_utterances
            
    def embed_sentences (
        sentences : List[str] , 
        model : SentenceTransformer
        ) -> List[Union[np.array , List[float]]] : 
        """
        This function takes in a list of sentences and returns their embeddings using the OpenAI Ada-002 model.

        :param sentences: List of sentences to be embedded
        :return: List of embeddings for each sentence
        """
        # Make the API call to get the embeddings
        embeddings = model.encode(sentences=sentences)
        
        
        return embeddings
    
    def embed_sampled_data(sampled_data : Dict , model : SentenceTransformer) : 
        data = {}
        for i, key in tqdm(enumerate(sampled_data.keys(), 1) , desc = "embedding in progress ..."):
            sentences = sampled_data[key]
            embeddings = ExtractEmbed.embed_sentences(sentences , model )
            data[key] = [
                {"utterance": sentence, "embedding": list(embedding)}
                for sentence, embedding in zip(sentences, embeddings)
            ]
        return data 
            
    def extract_embeddings(data : Dict) : 
        # Extract embeddings
        all_embeddings = []
        for key in data:
            for item in data[key]:
                all_embeddings.append(item["embedding"])
        return all_embeddings