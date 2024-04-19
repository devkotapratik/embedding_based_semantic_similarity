# Embedding Based Semantic Similarity

## Improving the evaluation of NLP approaches for scientific text annotation with Ontology Embedding-Based Semantic Similarity Metrics

Ontologies are extensively used across various scientific domains, including biology, physics, and geography, to present data. Central to this representation is the curation and annotation of ontologies, wherein words and phrases in scientific texts are associated with appropriate ontology concepts. This process helps in providing semantic understanding that allows Natural Language Processing (NLP) systems to better comprehend the underlying meanings behind these texts. Consequently, it facilitates improved knowledge presentation, data integration, and information retrieval. While historically reliant on manual curation by human annotators, recent advancements in deep learning algorithms has led to the automation of these annotations, achieving accuracy close to human annotators.

To assess the efficacy of deep learning models, evaluation metrics like precision and recall are generally used. However, these conventional metrics are inadequate for evaluating the performance of ontology annotation models. This limitation is because of the interdependence of outputs on multiple ontology concepts, and thus the inability of treating the outputs as independent entities. Semantic similarity metrics are therefore considered a preferred alternative, since they offer the capacity to estimate partial accuracy and provide a more nuanced evaluation of models tailored for ontology annotations.

Here, we present a novel approach to yield a robust semantic similarity metrics, through the utilization of ontology embeddings generated from ontology hierarchy, offering a more robust and accurate assessment of NLP approaches in the domain of ontology annotations. This repository houses all the source codes pertaining to the methods elucidated in our recent publication [Improving the Evaluation of NLP Approaches for Scientific Text Annotation with Ontology Embedding-Based Semantic Similarity Metrics](https://github.com/devkotapratik/embedding_based_semantic_similarity). This repository also serves as a valuable resource for researchers and practitioners interested in replicating and verifying the results detailed in the paper. By providing access to th codebase, we aim to enhance the reproducibility of the findings presented in the study. Additionally, the repository offers a platform for collaborative engagement and further exploration of the proposed methods in the field of onology annotations and NLP.

You'll find the following in this repositoy:
* `data/` - directory for data including dataset and embedding vectors
    * `annotations/`
        * `UBERON/` - contains dataset independently annotated by three experts for the Phenoscape Project and also the Gold Standard dataset (`GS_Dataset.tsv`)
* `scripts/` - scripts to generate UBERON ontology embeddings, compute semantic similarity and combine top metrics to develop a more robust "super" metric
* `README.md` - contains project description
* `requirements.txt` - contains list of dependencies required to run the project


## Hardware and Software Requirements

This codebase has been tested on Python 3.7.17 with NVIDIA A6000 GPUs on linux operating system. The codebase also utilizes distributed GPU computation wherever possible.


## Installation

It is assumed that you have already installed python. If not, you can install python following the [official python documentation](https://docs.python.org/3.7/using/index.html). To get started, clone the repositoy and setup your environment. We **strongly** recommend working on a virtual environment.

```bash
git clone https://github.com/devkotapratik/embedding_based_semantic_similarity.git
cd embedding_based_semantic_similarity

# Creating and activating a virtual environment
python3 -m venv uberon_venv
source uberon_venv/bin/activate

pip install -r requirements.txt
```