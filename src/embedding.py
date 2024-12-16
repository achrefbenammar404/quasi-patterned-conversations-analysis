from typing import List  , Dict , Union , Any
import numpy as np 
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 
from src.utils.utils import read_json_to_dict , save_dict_to_json
import os 

cache = read_json_to_dict(os.path.join("data" , "embedding_cache.json"))

update_cache = False 

class ExtractEmbed : 
    
    def extract_utterances(
        processed_formatted_conversations : dict 
    ) -> List[List[str]] : 
        customer_support_agent_utterances = []
        for i , conv in enumerate(processed_formatted_conversations.values()) : 
            customer_support_agent_utterances.append([
                utterance["content"] for utterance in conv if utterance["role"] in["agent"]
            ]) 
        return customer_support_agent_utterances
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
                utterance["content"] for utterance in conv if utterance["role"] in["agent"]
            ]
        return customer_support_agent_utterances
            
    def embed_sentences (
        sentences : List[str] , 
        model : SentenceTransformer , 
        dataset_name : str 
        ) -> List[Union[np.array , List[float]]] : 
        """This function takes in a list of sentences and returns their embeddings using the OpenAI Ada-002 model.


        Args:
            sentences (List[str]): List of sentences to be embedded
            model (SentenceTransformer): Hugging face model for embedding

        Returns:
            List[Union[np.array , List[float]]]: list of embeddings 
        """
        # Make the API call to get the embeddings
        embeddings = model.encode(sentences=sentences)
        return embeddings
    
    def embed_sampled_data(sampled_data : Dict , model : SentenceTransformer , dataset_name) -> Dict[str , Any] : 
        global update_cache
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
            if f"dataset_name_{key}" in cache : 
                embeddings = np.array(cache[f"dataset_name_{key}"])
            else :
                update_cache = True 
                embeddings = ExtractEmbed.embed_sentences(sentences , model  , dataset_name )
                cache[f"dataset_name_{key}"] = embeddings 
            data[key] = [
                    {"utterance": sentence, "embedding": list(embedding)}
                    for sentence, embedding in zip(sentences, embeddings)
            ]
        if update_cache : 
            save_dict_to_json( cache ,os.path.join("data" , "embedding_cache.json") )
            
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