# ============================================================
# intent_classifier/dataset_builder.py
# PURPOSE: Generate + curate training data for the classifier
# ============================================================

import json
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple
from pathlib import Path


@dataclass
class TrainingExample:
    """Single training example for intent classification."""
    text: str
    label: str
    source: str  # 'template', 'augmented', 'manual', 'scraped'
    confidence: float = 1.0  # How confident we are in this label


class IntentDatasetBuilder:
    """
    Builds a training dataset for intent classification.
    
    WHY BUILD OUR OWN?
    - No perfect dataset exists for this exact 5-class problem
    - We need domain-specific examples
    - Shows recruiters you understand data engineering
    - Template + augmentation approach is production-standard
    """
    
    # Seed templates — these capture the PATTERNS of each intent
    TEMPLATES = {
        'advice': [
            "How should I {action} when {situation}?",
            "What's the best way to {action}?",
            "I need help with {topic}, what do you recommend?",
            "Tips for {action} as a {role}?",
            "Should I {option_a} or {option_b}?",
            "What would you suggest for {situation}?",
            "How do I deal with {problem}?",
            "Best practices for {topic}?",
            "I'm struggling with {problem}, any advice?",
            "What's your opinion on {topic}?",
            "How can I improve my {skill}?",
            "What approach works best for {situation}?",
        ],
        'history': [
            "What happened during {event}?",
            "When did {event} occur?",
            "Who was {person} and what did they do?",
            "What caused {event}?",
            "Tell me about the history of {topic}",
            "How did {event} change {outcome}?",
            "What were the consequences of {event}?",
            "Who were the key figures in {event}?",
            "What led to the {event}?",
            "Describe the {era} period in {place}",
            "How did {civilization} rise and fall?",
            "What was life like during {era}?",
        ],
        'science': [
            "How does {phenomenon} work?",
            "What is the science behind {topic}?",
            "Explain {concept} in simple terms",
            "Why does {phenomenon} happen?",
            "What causes {phenomenon}?",
            "How is {substance} formed?",
            "What's the difference between {concept_a} and {concept_b}?",
            "Can you explain the theory of {theory}?",
            "What role does {element} play in {process}?",
            "How do scientists measure {phenomenon}?",
            "What are the properties of {substance}?",
            "How does {organ} function in the body?",
        ],
        'fact': [
            "What is {thing}?",
            "How many {thing} are there in {place}?",
            "What is the {superlative} {thing} in the world?",
            "Who invented {invention}?",
            "What is the capital of {place}?",
            "How tall is {structure}?",
            "What year was {thing} {action}?",
            "Define {term}",
            "What does {acronym} stand for?",
            "How much does {thing} weigh?",
            "Where is {place} located?",
            "What is the population of {place}?",
        ],
        'verify': [
            "Is it true that {claim}?",
            "Can you fact-check: {claim}",
            "I heard that {claim}, is this accurate?",
            "True or false: {claim}",
            "Is {claim} a myth or reality?",
            "Does {thing} really {action}?",
            "Someone told me {claim}, verify this",
            "Debunk or confirm: {claim}",
            "Is there evidence that {claim}?",
            "Is the claim that {claim} supported by science?",
            "Myth or fact: {claim}?",
            "Has it been proven that {claim}?",
        ]
    }
    
    # Fill-in values for template expansion
    FILL_VALUES = {
        'action': ['negotiate a salary', 'learn programming', 'start investing',
                    'lose weight', 'build muscle', 'write a resume', 'manage time',
                    'cook healthy meals', 'save money', 'study effectively'],
        'situation': ['starting a new job', 'dealing with conflict', 'planning a career change',
                      'moving to a new city', 'going through a breakup', 'starting college'],
        'topic': ['machine learning', 'quantum physics', 'the Roman Empire',
                  'cryptocurrency', 'climate change', 'mental health',
                  'artificial intelligence', 'renewable energy', 'space exploration'],
        'problem': ['procrastination', 'anxiety', 'debt', 'insomnia',
                    'lack of motivation', 'burnout', 'loneliness'],
        'event': ['World War II', 'the French Revolution', 'the Moon landing',
                  'the fall of the Berlin Wall', 'the Renaissance',
                  'the Industrial Revolution', 'the Cuban Missile Crisis'],
        'person': ['Napoleon', 'Cleopatra', 'Einstein', 'Ada Lovelace',
                   'Genghis Khan', 'Marie Curie', 'Alexander the Great'],
        'phenomenon': ['lightning', 'black holes', 'photosynthesis',
                       'magnetism', 'gravity', 'evolution', 'nuclear fusion'],
        'concept': ['entropy', 'natural selection', 'relativity',
                    'quantum entanglement', 'DNA replication', 'plate tectonics'],
        'thing': ['the Eiffel Tower', 'Bitcoin', 'the human brain',
                  'the Amazon River', 'the International Space Station'],
        'place': ['Japan', 'Antarctica', 'the Sahara Desert', 'Mars', 'Iceland'],
        'claim': ['humans only use 10% of their brain',
                  'the Great Wall of China is visible from space',
                  'goldfish have a 3-second memory',
                  'lightning never strikes the same place twice',
                  'eating carrots improves your vision',
                  'cracking knuckles causes arthritis',
                  'we swallow 8 spiders per year in our sleep'],
        'role': ['beginner', 'student', 'professional', 'parent', 'teenager'],
        'skill': ['public speaking', 'writing', 'cooking', 'coding', 'drawing'],
        'superlative': ['tallest', 'largest', 'oldest', 'deepest', 'fastest'],
        'substance': ['water', 'diamond', 'steel', 'plastic', 'glass'],
        'era': ['Medieval', 'Victorian', 'Bronze Age', 'Renaissance', 'Cold War'],
        'civilization': ['the Roman Empire', 'Ancient Egypt', 'the Aztec Empire',
                         'the Ottoman Empire', 'the Ming Dynasty'],
        'theory': ['evolution', 'relativity', 'the Big Bang', 'plate tectonics'],
        'term': ['democracy', 'entropy', 'capitalism', 'photosynthesis'],
        'structure': ['the Eiffel Tower', 'Mount Everest', 'the Burj Khalifa'],
        'invention': ['the telephone', 'the lightbulb', 'the internet', 'penicillin'],
        'acronym': ['NASA', 'NATO', 'UNESCO', 'CERN', 'WHO'],
        'option_a': ['save money', 'go to college', 'rent', 'learn Python'],
        'option_b': ['invest early', 'learn a trade', 'buy a house', 'learn JavaScript'],
        'outcome': ['modern society', 'world politics', 'technology', 'culture'],
        'element': ['oxygen', 'carbon', 'iron', 'nitrogen', 'calcium'],
        'process': ['respiration', 'digestion', 'combustion', 'fermentation'],
        'organ': ['the heart', 'the liver', 'the brain', 'the kidney'],
    }
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.examples: List[TrainingExample] = []
        random.seed(seed)
    
    def _fill_template(self, template: str) -> str:
        """Replace {placeholders} with random values."""
        result = template
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        for ph in placeholders:
            # Handle variations like concept_a, concept_b
            base_key = ph.rstrip('_abcdefg').rstrip('_')
            if base_key in self.FILL_VALUES:
                values = self.FILL_VALUES[base_key]
                result = result.replace('{' + ph + '}', random.choice(values), 1)
            elif ph in self.FILL_VALUES:
                result = result.replace('{' + ph + '}', random.choice(self.FILL_VALUES[ph]), 1)
        return result
    
    def generate_from_templates(self, examples_per_template: int = 10) -> None:
        """Generate training examples by filling templates."""
        for label, templates in self.TEMPLATES.items():
            for template in templates:
                for _ in range(examples_per_template):
                    filled = self._fill_template(template)
                    self.examples.append(TrainingExample(
                        text=filled,
                        label=label,
                        source='template'
                    ))
        print(f"Generated {len(self.examples)} template examples")
    
    def augment_with_noise(self, noise_ratio: float = 0.3) -> None:
        """
        Data augmentation techniques:
        - Typo injection (real users make typos)
        - Case variation
        - Synonym replacement (basic)
        - Word dropping
        
        WHY: Makes the model robust to real-world input
        """
        augmented = []
        original_count = len(self.examples)
        
        for example in self.examples[:]:  # Copy to avoid infinite loop
            if random.random() > noise_ratio:
                continue
            
            text = example.text
            augmentation_type = random.choice(['typo', 'case', 'drop_word'])
            
            if augmentation_type == 'typo':
                # Swap two adjacent characters
                if len(text) > 3:
                    idx = random.randint(1, len(text) - 2)
                    text = text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
            
            elif augmentation_type == 'case':
                # Random case changes
                text = text.lower() if random.random() > 0.5 else text.upper()
            
            elif augmentation_type == 'drop_word':
                # Remove a random word (simulates incomplete queries)
                words = text.split()
                if len(words) > 3:
                    drop_idx = random.randint(1, len(words) - 1)
                    words.pop(drop_idx)
                    text = ' '.join(words)
            
            augmented.append(TrainingExample(
                text=text,
                label=example.label,
                source='augmented',
                confidence=0.9  # Slightly lower confidence for augmented data
            ))
        
        self.examples.extend(augmented)
        print(f"Augmented: {original_count} → {len(self.examples)} examples")
    
    def add_hard_negatives(self) -> None:
        """
        Add examples that are INTENTIONALLY tricky.
        These are cases where the mode is ambiguous.
        
        WHY: This is what separates a toy project from real ML engineering.
        Interviewers LOVE hearing about hard negative mining.
        """
        hard_negatives = [
            # Could be ADVICE or FACT — context matters
            TrainingExample("What's the best programming language?", "advice", "manual"),
            TrainingExample("What is Python used for?", "fact", "manual"),
            
            # Could be HISTORY or FACT
            TrainingExample("When was America founded?", "fact", "manual"),
            TrainingExample("How was America founded?", "history", "manual"),
            
            # Could be SCIENCE or VERIFY
            TrainingExample("Does vitamin C cure colds?", "verify", "manual"),
            TrainingExample("How does vitamin C affect the immune system?", "science", "manual"),
            
            # Could be VERIFY or FACT
            TrainingExample("Is Pluto a planet?", "verify", "manual"),
            TrainingExample("What classification is Pluto?", "fact", "manual"),
            
            # Conversational / ambiguous phrasing
            TrainingExample("yo what even is dark matter lol", "science", "manual"),
            TrainingExample("bro is it true that we only use 10% of brain", "verify", "manual"),
            TrainingExample("help me understand the stock market", "advice", "manual"),
            TrainingExample("did napoleon actually lose at waterloo", "verify", "manual"),
            TrainingExample("tell me about waterloo", "history", "manual"),
            TrainingExample("what happened at waterloo", "history", "manual"),
        ]
        
        self.examples.extend(hard_negatives)
        print(f"Added {len(hard_negatives)} hard negative examples")
    
    def get_dataset_stats(self) -> Dict:
        """Return dataset statistics."""
        from collections import Counter
        label_counts = Counter(ex.label for ex in self.examples)
        source_counts = Counter(ex.source for ex in self.examples)
        return {
            'total': len(self.examples),
            'by_label': dict(label_counts),
            'by_source': dict(source_counts),
            'avg_text_length': sum(len(ex.text) for ex in self.examples) / len(self.examples)
        }
    
    def save(self, filepath: str = 'data/intent_dataset.json') -> None:
        """Save dataset to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(ex) for ex in self.examples]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.examples)} examples to {filepath}")
    
    def build_full_dataset(self) -> List[TrainingExample]:
        """Run the complete dataset building pipeline."""
        print("=" * 50)
        print("Building Intent Classification Dataset")
        print("=" * 50)
        
        self.generate_from_templates(examples_per_template=15)
        self.augment_with_noise(noise_ratio=0.3)
        self.add_hard_negatives()
        
        stats = self.get_dataset_stats()
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        self.save()
        return self.examples


# Build the dataset
if __name__ == '__main__':
    builder = IntentDatasetBuilder(seed=42)
    examples = builder.build_full_dataset()