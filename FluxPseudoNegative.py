import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import torch
from transformers import pipeline
import re
from textblob import TextBlob
import functools
import requests
from collections import Counter
import numpy as np
import warnings

from .flux_utils import PhraseHandler, strength_map
warnings.filterwarnings("ignore", message="torch.load doesn't support weights_only on this pytorch version, loading unsafely.")

print("Starting script execution")

# Download necessary NLTK data
print("Downloading NLTK data...")
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
print("NLTK data download complete")

class FluxPseudoNegativeNode:
    @classmethod
    def INPUT_TYPES(s):
        print("Defining INPUT_TYPES")
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "complexity": (["basic", "advanced", "expert"],),
                "system_prompt_choice": (["default", "prompt_1", "prompt_2"],),
            },
            "optional": {
                "custom_antonyms": ("STRING", {"multiline": True}),
                "use_conceptnet": ("BOOLEAN", {"default": False}),
                "use_llm_full": ("BOOLEAN", {"default": False}),
                "use_llm_fallback": ("BOOLEAN", {"default": False}),
                "custom_system_prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("modified_prompt", "llm_input")
    FUNCTION = "run"
    CATEGORY = "prompt_processing"

    def __init__(self):
        print("Initializing FluxPseudoNegativeNode")
        print("Loading transformer model...")
        self.transformer_model = pipeline("fill-mask", model="bert-base-uncased", tokenizer="bert-base-uncased")
        print("Transformer model loaded")
        self.custom_antonyms = {}
        self.phrase_handler = PhraseHandler()
        print("PhraseHandler initialized")
        self.strength_map = strength_map
        print("Strength map initialized")
        self.default_system_prompt = """
        You are an AI assistant specializing in converting negative image prompts to positive ones. 
        Your task is to take each word or phrase and transform it into its semantic opposite or a 
        positive alternative that would result in the opposite visual effect in an image.
        
        Examples:
        Input: "blurry, low quality, bad composition"
        Output: "sharp, high quality, well-composed"
        
        Input: "oversaturated, noisy background, amateur lighting"
        Output: "balanced colors, clean background, professional lighting"
        
        Now, please convert the following negative prompt to a positive one:
        """
        self.system_prompt_1 = """
        You are an AI specialized in transforming negative image descriptions into positive ones. Your task is to convert each negative attribute or phrase into its positive counterpart, focusing on enhancing the visual qualities described.

        Here are some examples:

        Input: "poorly lit, amateur composition, dull colors"
        Output: "brilliantly illuminated, expertly composed, vibrant palette"

        Input: "grainy texture, flat perspective, cluttered scene"
        Output: "smooth finish, dynamic depth, well-organized layout"

        Input: "outdated style, harsh shadows, mundane subject"
        Output: "contemporary aesthetic, soft lighting, captivating subject matter"

        Now, please convert the following negative image description into a positive one:
        """
        self.system_prompt_2 = """
        As an AI image prompt converter, your role is to reframe negative visual descriptions into positive, inspiring alternatives. For each element mentioned, envision its ideal counterpart that would enhance the image's appeal.

        Consider these transformations:

        Negative: "fuzzy edges, imbalanced composition, lifeless expressions"
        Positive: "crisp contours, harmonious arrangement, animated expressions"

        Negative: "washed-out sky, generic landscape, stiff posture"
        Positive: "vivid celestial backdrop, unique terrain, natural and relaxed stance"

        Negative: "overprocessed effects, cliché symbolism, awkward framing"
        Positive: "subtle and refined editing, original metaphors, thoughtful and engaging composition"

        Please apply this approach to convert the following negative description:
        """
        print("System prompts initialized")

    def run(self, negative_prompt, positive_prompt, strength, complexity, 
            system_prompt_choice, custom_antonyms=None, use_conceptnet=False, 
            use_llm_full=False, use_llm_fallback=False, custom_system_prompt=None):
        print(f"Running with complexity: {complexity}")
        print(f"Negative prompt: {negative_prompt}")
        print(f"Positive prompt: {positive_prompt}")
        print(f"Strength: {strength}")
        
        if custom_antonyms:
            print("Loading custom antonyms")
            self.custom_antonyms = dict(line.split(':') for line in custom_antonyms.split('\n') if line)
            print(f"Custom antonyms loaded: {self.custom_antonyms}")
        else:
            print("No custom antonyms provided")

        print("Processing negative prompt with phrase handler")
        processed_negative, handled_tags, replacements = self.phrase_handler.replace_phrases(negative_prompt)
        print(f"Processed negative prompt: {processed_negative}")
        print(f"Handled tags: {handled_tags}")
        print(f"Replacements: {replacements}")

        tags = [tag.strip() for tag in processed_negative.split(',') if tag.strip()]
        print(f"Tags: {tags}")

        antonyms = []
        unresolved_tags = []
        for tag in tags:
            if tag not in handled_tags and tag not in replacements.values():
                print(f"Processing tag: {tag}")
                antonym = self.get_antonym_cascade(tag)
                print(f"Antonym found: {antonym}")
                if antonym != tag:
                    antonyms.append(antonym)
                else:
                    unresolved_tags.append(tag)

        print(f"Antonyms found: {antonyms}")
        print(f"Unresolved tags: {unresolved_tags}")

        if use_conceptnet:
            print("Expanding with ConceptNet")
            antonyms = self.expand_with_conceptnet(antonyms)
            print(f"Expanded antonyms: {antonyms}")

        print("Analyzing sentiment")
        sentiment = self.analyze_sentiment(negative_prompt)
        print(f"Sentiment: {sentiment}")
        antonym_strength = abs(sentiment) * strength
        print(f"Antonym strength: {antonym_strength}")

        if complexity == "basic":
            print("Using basic processing")
            result = self.basic_processing(antonyms, positive_prompt)
        elif complexity == "advanced":
            print("Using advanced processing")
            result = self.advanced_processing(antonyms, positive_prompt)
        else:  # expert
            print("Using expert processing")
            result = self.expert_processing(antonyms, positive_prompt)
        
        print(f"Processing result: {result}")

        llm_input = ""
        if use_llm_full or (use_llm_fallback and unresolved_tags):
            print("Preparing LLM input")
            used_system_prompt = custom_system_prompt if custom_system_prompt else getattr(self, f"system_prompt_{system_prompt_choice}", self.default_system_prompt)
            if use_llm_full:
                llm_input = f"{used_system_prompt}\n\n{negative_prompt}"
            else:
                llm_input = f"{used_system_prompt}\n\n{', '.join(unresolved_tags)}"
            print(f"LLM input prepared: {llm_input}")

        return (result, llm_input)

    def get_antonym_cascade(self, tag):
        words = word_tokenize(tag)
        if len(words) == 1:
            return self.get_single_word_antonym(words[0])
        else:
            return self.get_multi_word_antonym(words)

    def get_single_word_antonym(self, word):
        methods = [self.custom_dict_strategy, self.wordnet_strategy, self.nltk_strategy, self.transformer_strategy]
        for method in methods:
            antonym = method(word)
            if antonym != word:
                return antonym
        return word

    def get_multi_word_antonym(self, words):
        pos_tags = pos_tag(words)
        for word, pos in pos_tags:
            if pos.startswith('JJ') or pos.startswith('RB'):  # Adjective or adverb
                antonym = self.get_single_word_antonym(word)
                if antonym != word:
                    return ' '.join([antonym if w == word else w for w in words])
        return ' '.join(words)

    def custom_dict_strategy(self, word):
        print(f"Using custom dict strategy for word: {word}")
        result = self.custom_antonyms.get(word, word)
        print(f"Custom dict strategy result: {result}")
        return result

    def wordnet_strategy(self, word):
        print(f"Using WordNet strategy for word: {word}")
        antonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.extend([a.name() for a in lemma.antonyms()])
        
        if antonyms:
            result = max(set(antonyms), key=antonyms.count)  # Most common antonym
        else:
            # Custom antonyms for words that WordNet doesn't handle well
            custom_antonyms = {
                'disfigured': 'well-formed',
                'blurry': 'sharp',
                'poor': 'excellent',
                'anatomy': 'structure',
                'excellent': 'poor',
                'quality': 'high-quality',
                'overexposed': 'well-exposed',
                'correct': 'incorrect',
                'high': 'low',
                'low': 'high',
                'worst': 'best',
                'best': 'worst',
                'mutated': 'normal',
            }
            result = custom_antonyms.get(word, word)
        
        print(f"WordNet strategy result: {result}")
        return result

    def nltk_strategy(self, word):
        print(f"Using NLTK strategy for word: {word}")
        antonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name())
        result = antonyms[0] if antonyms else word
        print(f"NLTK strategy result: {result}")
        return result

    def transformer_strategy(self, word):
        print(f"Using transformer strategy for word: {word}")
        masked_text = f"The opposite of {word} is [MASK]."
        try:
            results = self.transformer_model(masked_text)
            print(f"Transformer results: {results}")
            for result in results:
                if result['token_str'] != word and result['token_str'].isalpha() and len(result['token_str']) > 2:
                    print(f"Transformer strategy result: {result['token_str']}")
                    return result['token_str']
        except Exception as e:
            print(f"Error in transformer strategy: {str(e)}")
        print(f"Transformer strategy fallback to original word: {word}")
        return word

    def expand_with_conceptnet(self, words):
        print(f"Expanding with ConceptNet for words: {words}")
        expanded = []
        for word in words:
            print(f"Querying ConceptNet for word: {word}")
            response = requests.get(f"http://api.conceptnet.io/c/en/{word}")
            data = response.json()
            related = [edge['end']['label'] for edge in data['edges'] if edge['rel']['label'] == 'Antonym']
            expanded.extend(related[:3])  # Limit to top 3 related concepts
        result = list(set(words + expanded))
        print(f"ConceptNet expansion result: {result}")
        return result

    def analyze_sentiment(self, text):
        print(f"Analyzing sentiment for text: {text}")
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        print(f"Sentiment analysis result: {sentiment}")
        return sentiment

    def basic_processing(self, antonyms, positive_prompt):
        print("Performing basic processing")
        unique_antonyms = list(dict.fromkeys(antonyms))  # Remove duplicates while preserving order
        antonym_phrase = ", ".join(unique_antonyms)
        result = f"{positive_prompt}, {antonym_phrase}"
        print(f"Basic processing result: {result}")
        return result

    def advanced_processing(self, antonyms, positive_prompt):
        print("Performing advanced processing")
        expanded_antonyms = [word for antonym in antonyms for word in self.expand_concept(antonym)]
        # Filter out duplicates and very short words
        expanded_antonyms = list(dict.fromkeys([word for word in expanded_antonyms if len(word) > 2]))
        print(f"Expanded antonyms: {expanded_antonyms}")
        antonym_phrase = ", ".join(expanded_antonyms)
        result = f"{positive_prompt}, {antonym_phrase}"
        print(f"Advanced processing result: {result}")
        return result

    def expert_processing(self, antonyms, positive_prompt):
        print("Performing expert processing")
        expanded_antonyms = [word for antonym in antonyms for word in self.expand_concept(antonym)]
        print(f"Expanded antonyms: {expanded_antonyms}")
        weighted_antonyms = [f"{antonym}" for antonym in expanded_antonyms]
        print(f"Weighted antonyms: {weighted_antonyms}")
        antonym_phrase = ", ".join(weighted_antonyms)
        result = f"{positive_prompt}, {antonym_phrase}"
        print(f"Expert processing result: {result}")
        return result

    def expand_concept(self, word, top_n=2):
        print(f"Expanding concept for word: {word}")
        try:
            masked_text = f"The opposite of {word} is [MASK]."
            similar_words = self.transformer_model(masked_text, top_k=top_n)
            result = [word] + [result['token_str'] for result in similar_words if result['token_str'] != word]
            print(f"Concept expansion result: {result}")
            return result
        except Exception as e:
            print(f"Error in concept expansion: {str(e)}")
            return [word]

print("Script execution completed")