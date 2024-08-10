# FluxPseudoNegative: A ComfyUI Node for Converting Negative Prompts to Positive Attributes

FluxPseudoNegative is an advanced custom node for ComfyUI that converts negative prompts into positive ones. It's designed to enhance prompt engineering for image generation models that don't natively support negative prompts or where using negative prompts significantly increases generation time.

## Features

- Multiple antonym-finding strategies:
  - Custom phrase handling
  - WordNet
  - NLTK
  - Hugging Face Transformers
- Comprehensive phrase handling for multi-word concepts
- Sentiment analysis for strength adjustment
- Concept expansion using word embeddings
- Optional ConceptNet integration for expanded antonyms
- Multiple processing complexity levels: basic, advanced, expert
- Optional LLM integration for unresolved terms or full prompt conversion
- User-customizable antonyms and system prompts

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:

[CODE]
git clone https://github.com/yourusername/ComfyUI-FluxPseudoNegativePrompt.git
[/CODE]

2. Install the required dependencies:

[CODE]
pip install nltk textblob requests numpy transformers torch
[/CODE]

3. Download the required NLTK data:

[CODE]
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
[/CODE]

## Usage

1. In the ComfyUI interface, look for the "Flux Pseudo Negative" node under the "prompt_processing" category.

2. Connect the node to your workflow:
   - Input your negative prompt
   - Input a positive prompt to augment
   - Adjust the strength parameter (0.0 to 1.0)
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
- `complexity`: The processing complexity level (basic, advanced, expert)
- `custom_antonyms`: Optional custom antonym mappings
- `use_conceptnet`: Enable ConceptNet integration for concept expansion
- `use_llm_full`: Enable full LLM-based prompt conversion
- `use_llm_fallback`: Enable LLM-based fallback for unresolved terms
- `custom_system_prompt`: Custom system prompt for LLM integration

## File Structure

- `__init__.py`: Initializes the node for ComfyUI
- `FluxPseudoNegative.py`: Contains the main `FluxPseudoNegativeNode` class
- `flux_utils.py`: Contains the `PhraseHandler` class and `strength_map`

## Customization

You can customize the phrase mappings and strength map by modifying the `flux_utils.py` file.

## Note

This node requires significant computational resources, especially when using advanced NLP features and models. Performance may vary based on your system capabilities and the complexity of the input prompts.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Disclaimer

This node uses various third-party APIs and models. Please ensure you comply with their respective terms of use and licensing agreements.
