from src.pipelines import ufd_pipeline
from src.utils.utils import read_json_to_dict
from sentence_transformers import SentenceTransformer
import os 
import numpy as np 
import pandas as pd
import os

model = SentenceTransformer(model_name_or_path='sentence-transformers/all-MiniLM-L12-v2', device ="cuda" )

#defining the search space 
data_paths = [os.path.join("data" , "ABCD.json") , os.path.join("data" , "multiwoz.json")]
tau_s = np.arange(0 , 0.8 , 0.1)
alpha_s = np.arange(0 , 3 , 1)
top_k = np.arange(0 , 4 , 1)
clusters = np.arange(15 , 30 , 3)


def extract_utterances(data : dict , target_role : str ) -> dict : 
    return {key : [utterance for utterance in conv if utterance["role"] == target_role] for key , conv in data.items()}

def split_data(coff , data ) : 
    total_len = len(data)

    train_data_source = {str(i) : data[str(i)] for i in range(int(coff *total_len))}
    test_data_source = {str(i) : data[str(i)] for i in range(int(coff *total_len) , total_len)}
    
    return train_data_source , test_data_source





# Define result_df as a dictionary to store the results
result_df = {
    "approach_name": [],
    "graph_filtering_alg": [],
    "tau": [],
    "alpha": [],
    "top_k": [],
    "num_clusters": [],
    "semantic_coverage": [],
    "coverage": [],
    "branching_factor": [],
    "delta_hyperbolicity": [],
    "num_cycles": []
}

# Directory to save the results
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
output_file = os.path.join(output_dir, "results.csv")

# Assume other parts like `data_paths`, `clusters`, `tau_s`, `ufd_pipeline`, etc., are defined
alg_name = 'threshold_graph_builder'

for data_path in data_paths:
    data = read_json_to_dict(data_path)
    data = extract_utterances(data, "agent")
    train_data, test_data = split_data(0.8, data=data)

    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    for num_clusters in clusters:
        for tau in tau_s:
            try:
                print(f"Running on dataset: {dataset_name}, num_clusters: {num_clusters}, tau: {tau}, alpha: None, top_k: None")
                graph, score = ufd_pipeline.run(
                    dataset_name=dataset_name,
                    train_data=train_data,
                    test_data=test_data,
                    min_clusters=num_clusters,
                    max_clusters=num_clusters,
                    model=model,
                    tau=tau
                )

                # Append results to result_df
                result_df["approach_name"].append("UFD")
                result_df["graph_filtering_alg"].append(alg_name)
                result_df["tau"].append(tau)
                result_df["alpha"].append(None)
                result_df["top_k"].append(None)
                result_df["num_clusters"].append(num_clusters)
                result_df["coverage"].append(score["semantic_scores"]["coverage"])
                result_df["semantic_coverage"].append(score["semantic_scores"]["semantic_coverage"])
                result_df["branching_factor"].append(score["structural_scores"]["branching_factor"])
                result_df["delta_hyperbolicity"].append(score["structural_scores"]["delta_hyperbolicity"])
                result_df["num_cycles"].append(score["structural_scores"]["num_cycles"])

                # Convert the result dictionary to a DataFrame
                df = pd.DataFrame(result_df)

                # Save the DataFrame to a CSV file
                df.to_csv(output_file, index=False)

            except Exception as e:
                print(f"Run of dataset: {dataset_name}, num_clusters: {num_clusters}, tau: {tau}, alpha: None, top_k: None has failed: {e}")
