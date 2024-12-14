from typing import List  , Dict , Union , Any
import numpy as np 
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 
from src.utils.utils import read_json_to_dict , save_dict_to_json


cache = read_json_to_dict("data/embedding_cache.json")

class ExtractEmbed : 
    def extract_customer_support_utterances(
        processed_formatted_conversations : dict
        ) ->  Dict: 
        """extracts customer support utterances from the processed formatted json file

        Args:
            processed_formatted_conversations (dict): json file conataining conversations with keys being "conv-id" values being a list of dicts with attributes being 'role' and 'content'

        Returns:
            Dict: _description_
        """
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
        """This function takes in a list of sentences and returns their embeddings using the OpenAI Ada-002 model.


        Args:
            sentences (List[str]): List of sentences to be embedded
            model (SentenceTransformer): Hugging face model for embedding

        Returns:
            List[Union[np.array , List[float]]]: list of embeddings 
        """
        # Make the API call to get the embeddings
        if str(sentences) in cache : 
            return cache[str(sentences)]
        embeddings = model.encode(sentences=sentences)
        cache[str(sentences)] = embeddings
        
        return embeddings
    
    def embed_sampled_data(sampled_data : Dict , model : SentenceTransformer) -> Dict[str , Any] : 
        """method to embed the sampled conversations 

        Args:
            sampled_data (Dict): Dict with keys being conv-id and values being a list of customer support agent utterances 
            model (SentenceTransformer): _description_

        Returns:
            Dict[str , Any]: a dict with keys being the 'conv-id' and value being list of dicts with attributes utterance and embedding of that utterance 
        """
        data = {}
        for i, key in tqdm(enumerate(sampled_data.keys(), 1) , desc = "embedding in progress ..." , total=len(sampled_data)):
            sentences = sampled_data[key]

            embeddings = ExtractEmbed.embed_sentences(sentences , model )
            data[key] = [
                {"utterance": sentence, "embedding": list(embedding)}
                for sentence, embedding in zip(sentences, embeddings)
            ]
        save_dict_to_json( cache , "data/embedding_cache.json" )
        return data 
            
    def extract_embeddings(data : Dict) -> List[Union[List[float] , np.array]] : 
        """method to extract embeddings 

        Args:
            data (Dict): conversations data with key being 'conv-id' and value being the list of dicts with attributes utterance and embedding 

        Returns:
            _type_: list of all embeddings 
        """
        all_embeddings = []
        for key in data:
            for item in data[key]:
                all_embeddings.append(item["embedding"])
        return all_embeddings