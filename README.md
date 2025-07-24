# HypotheSAEs for Mental Health Research

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/mrjunos/depression-reddit-cleaned)
[![Forked from](https://img.shields.io/badge/Forked%20from-rmovva/HypotheSAEs-blue.svg)](https://github.com/rmovva/HypotheSAEs)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains a specialized adaptation of the **HYPOTHESAES** framework, fine-tuned for generating interpretable hypotheses from mental health text data. This work reproduces the methodology of the original paper using a dataset of Reddit posts related to depression, aiming to uncover cognitive patterns predictive of depressive states.

## Overview

While large language models (LLMs) can generate hypotheses, they struggle to scale, and while neural networks can process large datasets, their features are often uninterpretable. **HYPOTHESAES** bridges this gap by using Sparse Autoencoders (SAEs) to discover interpretable features (neurons) in text data and then correlating them with a target variable.

This fork applies the HYPOTHESAES method to a new domain: mental health. The goal is to automatically discover interpretable relationships between language patterns in user-generated text and the presence of depression.

### How HypotheSAEs Work

The method follows a three-step process:

1.  **Train a Sparse Autoencoder (SAE)**: An SAE is trained on text embeddings from the dataset. This produces a large set of "neurons," where each neuron learns to activate on a specific, narrow semantic concept present in the data.

2.  **Identify Predictive Neurons**: Lasso regression is used to find the small subset of SAE neurons that are most predictive of the target variable (in this case, a post from the r/depression subreddit).

3.  **Generate Natural Language Interpretations**: An LLM is used to interpret the predictive neurons. By showing the LLM examples of texts that strongly activate a neuron versus those that don't, it generates a concise, natural language hypothesis explaining what the neuron represents.

The result is a set of human-readable, testable hypotheses about the linguistic and cognitive markers of depression.

## Adaptation for Mental Health Research

### Dataset

This project uses the **[Depression Reddit Cleaned](https://huggingface.co/datasets/mrjunos/depression-reddit-cleaned)** dataset from Hugging Face. It contains posts from the r/depression subreddit (labeled as 1) and various other subreddits (labeled as 0).

### Key Modification to prompt:

To guide the LLM towards generating hypotheses relevant to cognitive science and mental health, the original interpretation prompt was replaced. The new prompt instructs the model to focus on *how* people think, rather than just *what* they think about.

> All texts are expressions of cognitive patterns and mental states. Your task is to identify and describe features that capture HOW people think, process information, and manage mental states, rather than just WHAT they think about.
>
> Focus on cognitive processes such as:
> - Information processing patterns (analytical vs. intuitive, systematic vs. scattered)
> - Mental effort and cognitive load indicators (complexity, elaboration, fatigue signs)
> - Reasoning styles (logical, emotional, biased, flexible)
> - Metacognitive awareness (self-reflection, monitoring one's own thinking)
> - Cognitive control and regulation (emotional management, attention control)
> - Mental flexibility vs. rigidity (openness to change, fixed thinking patterns)
> - Cognitive biases and heuristics (shortcuts, systematic errors in thinking)
> - Pattern recognition and connection-making abilities
>
> Features should be formulated as descriptive statements that explain the cognitive mechanism being expressed through language patterns, word choice, sentence structure, reasoning flow, and conceptual organization.
>
> Examples:
> - "demonstrates analytical thinking through systematic breakdown of complex ideas into component parts"
> - "exhibits cognitive load through fragmented sentence structures and incomplete thoughts"
> - "shows metacognitive awareness by explicitly reflecting on and questioning one's own reasoning process"
> - "manifests cognitive flexibility by actively considering and integrating opposing viewpoints"
> - "reveals confirmation bias through selective attention to supporting evidence while dismissing contradictory information"

## Results: Top 20 Predictive Features for Depression

Below are the top 20 most predictive features (neurons) correlated with posts from the r/depression subreddit. The `correlation` indicates the relationship with the depression, and `best_f1` measures how well the feature's interpretation classifies texts.

| Neuron | Correlation | F1 Score | Best Interpretation |
|:-------|:-----------:|:--------:|:--------------------|
| 303    | 0.370       | 0.78     | demonstrates cognitive overload through detailed recounting of emotionally intense personal experiences, often accompanied by fragmented thoughts and a lack of clear resolution |
| 315    | 0.366       | 0.52     | demonstrates dismissive or reductive reasoning by trivializing or minimizing the concept of depression through casual language, slang, and humor |
| 131    | 0.346       | 0.76     | demonstrates metacognitive awareness by expressing inner conflict and self-reflection about one's emotional state and actions, often questioning personal beliefs or societal norms |
| 98     | 0.335       | 0.40     | uses colloquial and casual language to express emotional states, often with abbreviated or fragmented phrasing that conveys a dismissive or resigned attitude toward the concept of depression |
| 262    | 0.308       | 0.78     | expresses heightened metacognitive awareness by frequently describing and analyzing one's own physical sensations and emotional states in detail |
| 295    | 0.293       | 0.81     | demonstrates cognitive overload and emotional exhaustion through repetitive expressions of hopelessness, self-doubt, and perceived lack of personal value in life |
| 272    | 0.277       | 0.77     | expresses a sense of cognitive exhaustion and mental overload through repetitive phrasing, fragmented expressions of hopelessness, and a focus on the inability to endure or escape current circumstances |
| 62     | 0.258       | 0.79     | expresses metacognitive awareness by explicitly seeking strategies, tools, or advice to manage and understand their own anxiety or emotional state |
| 273    | 0.234       | 0.83     | demonstrates a pervasive focus on suicidal ideation and planning, characterized by detailed descriptions of methods, potential consequences, and emotional rationalizations for self-harm or death |
| 293    | 0.226       | 0.18     | demonstrates metacognitive awareness by explicitly reflecting on the effectiveness of therapeutic approaches and questioning personal progress in managing mental health challenges |
| 263    | 0.218       | 0.79     | exhibits cognitive load and mental fatigue through repetitive expressions of exhaustion, lack of motivation, and difficulty initiating or maintaining basic routines and tasks |
| 113    | 0.202       | 0.58     | explicitly describes a process of identifying, analyzing, or managing anxiety-related symptoms or triggers, often with a focus on self-awareness or seeking clarification about their experiences |
| 103    | 0.199       | 0.87     | explicitly describes persistent and recurring suicidal thoughts as a central theme of cognitive focus |
| 166    | 0.198       | 0.75     | expresses cognitive overload and emotional turmoil through repetitive and self-referential language focused on themes of death, worthlessness, and existential uncertainty |
| 25     | 0.194       | 0.82     | expresses cognitive overload through detailed descriptions of persistent physical and mental symptoms, such as fatigue, dissociation, and difficulty functioning in social or public settings |
| 194    | 0.194       | 0.63     | demonstrates metacognitive awareness by reflecting on personal growth, self-regulation, and strategies for managing emotional and mental states |
| 233    | 0.192       | 0.79     | demonstrates persistent ruminative thinking by repeatedly focusing on personal flaws, past mistakes, or feelings of being a burden to others |
| 248    | 0.191       | 0.81     | expresses existential and nihilistic reasoning by questioning the purpose and meaning of life, often emphasizing the futility of actions and the impermanence of existence |
| 257    | 0.188       | 0.75     | exhibits a high level of self-reflection and personal narrative construction, often detailing past experiences, emotional struggles, and their impact on identity and future concerns |
| 159    | 0.184       | 0.36     | demonstrates metacognitive awareness by explicitly monitoring and questioning one's own mental states, symptoms, and thought processes in detail |

## Interactive Dashboard

To better understand these features, this repository includes an interactive dashboard where you can explore each neuron, its interpretation, and the top activating texts from the dataset.

![Dashboard Demo](images/dash.gif)

## Getting Started

To run this project and explore the results yourself, follow these steps.

### Prerequisites
*   Python 3.8+
*   Pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/HypotheSAEs-mental-health.git
    cd HypotheSAEs-mental-health
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```


## Acknowledgments & Citation

This work is a direct adaptation of the original **HYPOTHESAES** paper and repository. All credit for the core methodology goes to the original authors.

*   **Original Paper:** [HYPOTHESAES: A Framework for Generating Interpretable Hypotheses from Language Models](https://arxiv.org/abs/2502.04382)
*   **Original GitHub:** [https://github.com/rmovva/HypotheSAEs](https://github.com/rmovva/HypotheSAEs)
*   **Project Website:** [https://hypothesaes.org/](https://hypothesaes.org/)

If you use this adaptation in your research, please cite both this repository and the original HYPOTHESAES paper.