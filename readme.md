# FluxPseudoNegative Node for ComfyUI

The FluxPseudoNegative node is an advanced custom node for ComfyUI that converts negative prompts into positive ones. It's designed to enhance prompt engineering for image generation models that don't natively support negative prompts or where using negative prompts significantly increases generation time.

## Features

- Multiple antonym-finding strategies:
  - WordNet
  - spaCy
  - Custom dictionary
  - Flair
  - AllenNLP
  - Hugging Face Transformers
- Comprehensive phrase handling for multi-word concepts
- Sentiment analysis for strength adjustment
- Concept expansion using word embeddings and ConceptNet
- Semantic similarity checks
- Efficient caching for improved performance
- Multiple processing complexity levels: basic, advanced, expert
- Nuanced strength adjustments based on word semantics
- LLM integration options:
  - Full prompt conversion
  - Fallback for unresolved terms
- User-customizable system prompt for LLM integration

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:

```git clone https://github.com/yourusername/comfyui-fluxpseudonegative.git```

2. Install the required dependencies:

```pip install nltk spacy gensim textblob flair allennlp requests numpy```

3. Download the required NLTK data:

4. Download the required spaCy model:

```python -m spacy download en_core_web_sm```

## Usage

1. In the ComfyUI interface, look for the "Flux Pseudo Negative" node under the "prompt_processing" category.

2. Connect the node to your workflow:
   - Input your negative prompt
   - (Optional) Input a positive prompt to augment
   - Adjust the strength parameter (0.0 to 1.0)
   - Choose the antonym-finding strategy
   - Select the processing complexity
   - (Optional) Provide custom antonyms in the format "word:antonym" (one per line)
   - (Optional) Enable ConceptNet integration
   - (Optional) Enable LLM integration (full or fallback)
   - (Optional) Provide a custom system prompt for LLM integration

3. The node will output:
   - A modified positive prompt incorporating the converted negative concepts
   - (If LLM integration is enabled) An LLM input string for further processing

## Parameters

- `negative_prompt`: The negative prompt to convert
- `positive_prompt`: An optional positive prompt to augment
- `strength`: The strength of the antonym influence (0.0 to 1.0)
- `strategy`: The antonym-finding strategy to use
- `complexity`: The processing complexity level
- `custom_antonyms`: Optional custom antonym mappings
- `use_conceptnet`: Enable ConceptNet integration for concept expansion
- `use_llm_full`: Enable full LLM-based prompt conversion
- `use_llm_fallback`: Enable LLM-based fallback for unresolved terms
- `system_prompt`: Custom system prompt for LLM integration

## Strategies

1. **WordNet**: Uses NLTK's WordNet to find antonyms based on synsets.
2. **spaCy**: Utilizes spaCy for context-aware antonym selection, considering word similarity and frequency.
3. **Custom Dictionary**: Allows users to define their own antonym mappings.
4. **Flair**: Uses Flair for part-of-speech aware antonym finding.
5. **AllenNLP**: Employs coreference resolution for context-aware antonym selection.
6. **Transformers**: Uses BERT to suggest contextually appropriate replacements.

## Complexity Levels

1. **Basic**: Simple antonym replacement with uniform strength.
2. **Advanced**: Includes concept expansion and applies uniform strength.
3. **Expert**: Employs concept expansion with individual strength adjustments for each antonym based on word semantics.

## Phrase Handling

The node includes a comprehensive phrase map for common expressions in image generation prompts, covering aspects such as image quality, composition, lighting, color, style, and more.

## Strength Adjustment

The node uses a sophisticated strength adjustment system that takes into account the semantics of words. It includes a predefined strength map for common words and can interpolate strengths for unknown words based on their similarity to known words.

## ConceptNet Integration

When enabled, the node queries the ConceptNet API to find related concepts and antonyms, expanding the range of positive concepts generated from the negative prompt.

## LLM Integration

The node offers two options for LLM integration:
1. Full prompt conversion: Sends the entire negative prompt to an LLM for conversion.
2. Fallback for unresolved terms: Sends only the terms that couldn't be resolved by other methods to an LLM.

Users can provide a custom system prompt to guide the LLM's behavior.

## Note

This node requires significant computational resources, especially when using advanced NLP features and models. Performance may vary based on your system capabilities and the complexity of the input prompts.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Disclaimer

This node uses various third-party APIs and models. Please ensure you comply with their respective terms of use and licensing agreements.