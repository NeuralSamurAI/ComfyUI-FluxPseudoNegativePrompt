# flux_utils.py
from nltk import ngrams, word_tokenize

class PhraseHandler:
    def __init__(self):
        self.phrase_map = {
            # Quality and resolution
            "low quality": "high quality",
            "bad quality": "excellent quality",
            "poor quality": "superior quality",
            "low resolution": "high resolution",
            "blurry image": "sharp image",
            "pixelated": "smooth",
            "grainy": "clear",
            "noisy": "clean",
            "artifacted": "artifact-free",
            "compressed": "uncompressed",
            "lossy": "lossless",

            # Composition and framing
            "bad composition": "well-composed",
            "poor framing": "well-framed",
            "unbalanced composition": "balanced composition",
            "cluttered composition": "clean composition",
            "awkward pose": "natural pose",
            "stiff pose": "relaxed pose",

            # Lighting and exposure
            "poorly lit": "well lit",
            "bad lighting": "excellent lighting",
            "harsh lighting": "soft lighting",
            "flat lighting": "dynamic lighting",
            "overexposed": "well-exposed",
            "underexposed": "properly exposed",
            "blown out highlights": "well-preserved highlights",
            "crushed shadows": "detailed shadows",

            # Color and contrast
            "washed out colors": "vibrant colors",
            "dull colors": "rich colors",
            "oversaturated": "naturally saturated",
            "desaturated": "colorful",
            "low contrast": "high contrast",
            "flat contrast": "dynamic contrast",
            "monochromatic": "colorful",

            # Focus and depth
            "out of focus": "in focus",
            "shallow depth of field": "deep depth of field",
            "flat image": "image with depth",

            # Style and aesthetics
            "ugly": "beautiful",
            "unattractive": "attractive",
            "plain": "visually interesting",
            "boring": "engaging",
            "generic": "unique",
            "amateur": "professional",
            "amateurish": "skillful",
            "unprofessional": "professional",
            "kitsch": "refined",
            "tacky": "elegant",
            "gaudy": "tasteful",

            # Artistic techniques
            "poorly drawn": "well drawn",
            "badly sketched": "skillfully sketched",
            "amateurish painting": "masterful painting",
            "rough brushstrokes": "refined brushstrokes",
            "sloppy linework": "precise linework",
            "unrefined": "polished",

            # Perspective and proportion
            "bad perspective": "correct perspective",
            "wonky perspective": "accurate perspective",
            "distorted proportions": "correct proportions",
            "unrealistic scale": "realistic scale",

            # Anatomy and figure
            "bad anatomy": "accurate anatomy",
            "incorrect anatomy": "correct anatomy",
            "deformed": "well-formed",
            "disproportionate": "proportionate",
            "asymmetrical face": "symmetrical face",
            "unnatural pose": "natural pose",

            # Texture and detail
            "lack of detail": "rich in detail",
            "over-simplified": "detailed",
            "flat textures": "realistic textures",
            "unrealistic skin": "lifelike skin",
            "plastic-looking": "natural-looking",

            # Environmental elements
            "dull background": "interesting background",
            "distracting background": "complementary background",
            "inconsistent lighting": "consistent lighting",
            "unrealistic shadows": "realistic shadows",
            "lack of atmosphere": "atmospheric",

            # Camera and lens effects
            "lens distortion": "undistorted",
            "chromatic aberration": "no chromatic aberration",
            "vignetting": "even exposure",
            "motion blur": "sharp and clear",

            # Digital artifacts
            "jpeg artifacts": "artifact-free",
            "banding": "smooth gradients",
            "moire patterns": "clean patterns",

            # Style-specific
            "uncanny valley": "photorealistic",
            "too cartoonish": "realistic",
            "overly stylized": "naturally styled",

            # Miscellaneous
            "unfinished": "complete",
            "rough draft": "polished final version",
            "lazy execution": "meticulously crafted",
            "uninspired": "creative",
            "derivative": "original",
            "cliché": "innovative",
            "inconsistent style": "consistent style",
            "mismatched elements": "harmonious elements",
            "poor use of space": "effective use of space",
            "lack of focal point": "clear focal point",
            "confusing layout": "intuitive layout",
            "jarring color scheme": "pleasing color scheme",
            "inappropriate tone": "appropriate tone",
            "lack of emotion": "emotionally evocative",
            "stiff": "dynamic",
            "lifeless": "vibrant",
            "fake-looking": "authentic-looking",
            "cheap-looking": "premium-looking",
            "dated": "timeless",
            "forgettable": "memorable",
        }
        print(f"Phrase map initialized with {len(self.phrase_map)} entries")

    def find_phrases(self, text):
        print(f"Finding phrases in text: {text}")
        tokens = word_tokenize(text)
        found_phrases = []
        for n in range(4, 0, -1):
            for gram in ngrams(tokens, n):
                phrase = " ".join(gram)
                if phrase in self.phrase_map:
                    found_phrases.append(phrase)
        print(f"Found phrases: {found_phrases}")
        return found_phrases

    def replace_phrases(self, text):
        print(f"Replacing phrases in text: {text}")
        phrases = self.find_phrases(text)
        handled_tags = set()
        replacements = {}
        for phrase in phrases:
            if phrase in text:
                replacement = self.phrase_map[phrase]
                text = text.replace(phrase, replacement)
                handled_tags.add(phrase)
                replacements[phrase] = replacement
        print(f"Text after phrase replacement: {text}")
        return text, handled_tags, replacements

strength_map = {
    # Extreme negatives
    "terrible": 1.0, "horrible": 0.98, "awful": 0.96, "dreadful": 0.94, "atrocious": 0.92,
    "abysmal": 0.90, "appalling": 0.88, "catastrophic": 0.86, "disastrous": 0.84,
    
    # Strong negatives
    "very bad": 0.82, "awful": 0.80, "poor": 0.78, "subpar": 0.76, "inferior": 0.74,
    "inadequate": 0.72, "unacceptable": 0.70, "disappointing": 0.68, "unsatisfactory": 0.66,
    
    # Moderate negatives
    "bad": 0.64, "flawed": 0.62, "deficient": 0.60, "problematic": 0.58, "questionable": 0.56,
    "lackluster": 0.54, "mediocre": 0.52, "so-so": 0.50, "average": 0.48,
    
    # Slight negatives
    "not great": 0.46, "unremarkable": 0.44, "ordinary": 0.42, "passable": 0.40, "tolerable": 0.38,
    
    # Neutral
    "okay": 0.36, "fair": 0.34, "decent": 0.32, "satisfactory": 0.30, "sufficient": 0.28,
    
    # Slight positives
    "rather good": 0.26, "pretty good": 0.24, "quite good": 0.22, "above average": 0.20,
    
    # Moderate positives
    "good": 0.18, "pleasing": 0.16, "quality": 0.14, "fine": 0.12, "commendable": 0.10,
    
    # Strong positives
    "very good": 0.08, "great": 0.06, "excellent": 0.04, "superb": 0.02, "outstanding": 0.00,
    
    # Extreme positives
    "perfect": 0.00, "flawless": 0.00, "impeccable": 0.00, "ideal": 0.00, "sublime": 0.00,
    
    # Frequency adverbs
    "never": 1.0, "rarely": 0.8, "seldom": 0.7, "occasionally": 0.6, "sometimes": 0.5,
    "often": 0.3, "usually": 0.2, "always": 0.0,
    
    # Quantity adjectives
    "no": 1.0, "few": 0.8, "some": 0.5, "many": 0.3, "most": 0.1, "all": 0.0,
    
    # Intensity adverbs
    "extremely": 0.9, "very": 0.7, "quite": 0.5, "fairly": 0.3, "slightly": 0.1,
    
    # Size and scale
    "tiny": 0.9, "small": 0.7, "medium": 0.5, "large": 0.3, "huge": 0.1,
    
    # Quality descriptors
    "low-quality": 0.8, "high-quality": 0.2, "inferior": 0.7, "superior": 0.3,
    "substandard": 0.75, "standard": 0.5, "premium": 0.25,
    
    # Completeness
    "incomplete": 0.8, "partial": 0.6, "mostly": 0.4, "nearly": 0.2, "complete": 0.0,
    
    # Clarity and focus
    "blurry": 0.8, "unclear": 0.7, "clear": 0.3, "sharp": 0.2, "crystal-clear": 0.1,
    
    # Composition
    "unbalanced": 0.7, "balanced": 0.3, "harmonious": 0.2,
    
    # Lighting
    "dark": 0.7, "dim": 0.6, "well-lit": 0.3, "bright": 0.2,
    
    # Color
    "monochrome": 0.6, "dull": 0.7, "vibrant": 0.3, "colorful": 0.2,
    
    # Texture
    "rough": 0.6, "smooth": 0.4, "silky": 0.2,
    
    # Style
    "plain": 0.6, "ordinary": 0.5, "stylish": 0.3, "elegant": 0.2,
    
    # Originality
    "cliche": 0.7, "unoriginal": 0.6, "original": 0.3, "innovative": 0.2,
    
    # Emotional impact
    "boring": 0.7, "interesting": 0.3, "captivating": 0.2, "mesmerizing": 0.1,
    
    # Skill level
    "amateurish": 0.8, "unprofessional": 0.7, "professional": 0.3, "masterful": 0.1,
    
    # Overall impression
    "unappealing": 0.8, "appealing": 0.2, "attractive": 0.1, "stunning": 0.0
}