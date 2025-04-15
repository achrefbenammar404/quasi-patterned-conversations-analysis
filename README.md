<h1 align="center">A Computational Approach to Modeling Conversational Systems</h1>
<h3 align="center">Analyzing Large-Scale Quasi-Patterned Dialogue Flows</h3>

<p align="center">
  <img src="https://utm.rnu.tn/utm/images/utm-og-img.png" alt="FST Logo" width="100"/>
  <img src="https://insat.rnu.tn/assets/images/logo_c.png" alt="INSAT logo" width="100">
  <p align = "center">
    <img src="https://ieeer8.org/wp-content/uploads/2022/03/IEEE-Region-8-Logo.png" alt="IEEE XPLORE logo" width="230">
  </p>
</p>


<p align="center">
  <strong>Official Implementation of the IEEE EUROCON 2025 Paper <br>A Computational Approach to Modeling Conversational Systems</h1>
Analyzing Large-Scale Quasi-Patterned Dialogue Flows</strong><br>
  <em>Mohamed Achref Ben Ammar</em> – National Institute of Applied Science and Technology (INSAT), University of Carthage, Tunisia<br>
  <em>Mohamed Taha Bennani</em> – University of Tunis El Manar (FST)
</p>

---

## Abstract

The rise of large language models (LLMs) has led to increasingly complex and loosely structured dialogues. In this work, we introduce a **computational graph-based framework** that models these quasi-patterned conversations. Central to our approach is the **Filter & Reconnect** method, a graph simplification technique that reduces conversational noise while preserving semantic structure.

Key outcomes:
- **2.06× improvement in semantic metric S** over prior methods
- **0 δ-hyperbolicity**, enforcing a tree-like, interpretable structure

This framework offers practical tools for monitoring and analyzing chatbot behavior, dialogue management systems, and user interaction patterns at scale.

---

## Methodology Overview

The methodology consists of the following core steps:

1. **Utterance Extraction**  
   Conversational utterances are extracted from a structured dataset consisting of multi-turn dialogues.

2. **Semantic Embedding**  
   Each utterance is transformed into a dense vector using a pre-trained text embedding model, capturing the semantic meaning of the message.

3. **Clustering of Intents**  
   Using hierarchical clustering techniques and a large language model (LLM), similar utterances are grouped together to identify key communicative intents.

4. **Markov Chain Construction**  
   A Markov Chain is built where nodes represent clustered intents and edges represent transitions between them in the dialogue flow.

5. **Graph Simplification: Filter & Reconnect**  
   The conversational graph undergoes a noise reduction process by removing irrelevant transitions while preserving semantic and structural coherence.

6. **Flow Pattern Analysis**  
   The resulting graph is then analyzed to identify quasi-patterned conversational flows, enabling improved interpretability and dialogue system evaluation.

---

## Setup

### 1. Install Dependencies

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/MacOS
venv\Scripts\activate           # Windows

# Install required packages
pip install -r requirements.txt

# Download required NLP model
python -m spacy download en_core_web_md
```

### 2. Create a `.env` File

At the project root, create a `.env` file and configure the following environment variables:

```dotenv
# Python setup
PYTHONPATH=${PYTHONPATH}:.

# Environment mode
ENVIRONMENT="local"

# API Keys
GOOGLE_API_KEY=
MISTRAL_API_KEY=
```

> Ensure your API keys are valid and have the appropriate access privileges.

---

## Input Data Format

This framework supports **ABCD v1.1**, **MultiWOZ 2.0**, or any **custom dataset** formatted as follows:

```json
{
  "conversation_1": [
    {"role": "agent", "content": "Hello, how can I help you today?"},
    {"role": "customer", "content": "I need assistance with my account."},
    {"role": "action", "content": "Agent opened account details."}
  ]
}
```

- Save your data file as: `data/processed_formatted_conversations.json`

---

## Run the Pipeline

```bash
python main.py \
    --file_path data/processed_formatted_conversations.json \
    --num_sampled_data 500 \
    --min_clusters 10 \
    --max_clusters 30 \
    --model_name 'sentence-transformers/all-mpnet-base-v2' \
    --label_model 'open-mixtral-8x22b' \
    --tau 0.15 \
    --top_k 2 \
    --alpha 0.8
```

---

## Advanced Configuration

| Parameter            | Description                                      | Default                         |
|----------------------|--------------------------------------------------|---------------------------------|
| `--num_sampled_data` | Number of conversations to sample                | 100                             |
| `--min_clusters`     | Minimum cluster count for elbow method           | 5                               |
| `--max_clusters`     | Maximum cluster count for elbow method           | 15                              |
| `--model_name`       | Sentence embedding model                         | 'all-MiniLM-L12-v2'             |
| `--label_model`      | LLM for labeling dialogue state clusters         | 'open-mixtral-8x22b'            |
| `--tau`              | Minimum transition probability threshold         | 0.1                             |
| `--top_k`            | Number of outgoing edges to retain per node      | 1                               |
| `--alpha`            | Balance between semantic similarity and topology | 1.0                             |

---

## Citation

If you use this codebase for your research, please cite:

```bibtex
@inproceedings{achref2025conversationalgraph,
  title={A Computational Approach to Modeling Conversational Systems: Analyzing Large-Scale Quasi-Patterned Dialogue Flows},
  author={Mohamed Achref Ben Ammar and Mohamed Taha Bennani},
  conference={IEEE EUROCON 2025 - The 21st International Conference on Smart Technologies},
  year={2025},
  publisher={IEEE},
}
```

---

## Contact

For questions, collaborations, or feedback, feel free to reach out:

- **Mohamed Achref Ben Ammar** – [mohamedachref.benammar@insat.ucar.tn](mailto:mohamedachref.benammar@insat.ucar.tn)  
- **Mohamed Taha Bennani** – [taha.bennani@fst.utm.tn](mailto:taha.bennani@fst.utm.tn)
