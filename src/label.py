from typing import Dict 
from src.llm import CollectionLLM
from src.utils.utils import pre_process_llm_output
import json 


class Label : 
    
    def generate_cluster_by_intent(intent_by_cluster : Dict ) : 
        return { value : key for key , value in intent_by_cluster.items()}
    def label_clusters_by_closest_utterances (
        closest_utterances : Dict  , 
        model : str) : 
        try : 
            client = CollectionLLM.llm_collection[model]
        except Exception as e : 
            print(f"error in label_clusters_by_closest_utterances , model is not included in CollectionLLM : {e}")
        intent_by_cluster = {}
        for cluster, utterances in closest_utterances.items():
            prompt = f"you will be given some utterances from a conversation between a customer support agent and clients from a specific company. you need to extract the intent of these utterances, your output is a simple short phrase describing the common overall intent of these utterances , replace any proper name or specification with the category of the object (for example when given the name Alice , you replace it with 'Customer Name'): \n"
            for idx , utterance in enumerate(utterances):
                prompt += "- " + utterance
            prompt += "\n your response should be a dict with one attribute that is 'intent' "
            completion = client.get_response(
                messages=[
                    {"role": "system", "content": "You are an AI agent specializing in intent and motive recoginition , you will be given utterances of a customer support agent from conversations with clients designed to output a json with one key 'intent'"},
                    {"role": "user", "content": prompt}
                ]

            )
            completion  = pre_process_llm_output(completion)
            try : 
                intent_by_cluster[str(cluster)] = completion["intent"]
            except Exception as e : 
                print(f"error reading completion response - completion response : {completion}")
                intent_by_cluster[str(cluster)] = completion
        return intent_by_cluster
                
    def add_intents_to_conversations(data, intents_by_cluster):
        for key in data:
            # Create an ordered list of intents for each conversation
            intents_ordered = []
            for item in data[key]:
                if 'ordered_intents' not in item.keys()  and len(item) != 0  : 
                    cluster = str(item['cluster'])
                    intent = intents_by_cluster.get(cluster, "Unknown")  # Handle missing intents
                    intents_ordered.append(intent)
            # Add the ordered list of intents to the conversation
            data[key].append({"ordered_intents": intents_ordered})
        return data
    
    def print_updated_data_with_ordered_intents(data : Dict ) : 
        for key in data:
            print(f"Conversation: {key}")
            for item in data[key]:
                if "utterance" in item:
                    print(f"  Utterance: {item['utterance']}, Cluster: {item['cluster']}")
                else:
                    print(f"  Ordered Intents: {item['ordered_intents']}")
                    
            
    def extract_ordered_intents(data : Dict ) : 
        ordered_intents = []
        for key in data:
            for item in data[key]:
                if "ordered_intents" in item:
                    ordered_intents.append(item["ordered_intents"])
        return ordered_intents