import collections
import re

class Evaluator:
    
    @staticmethod
    def normalize_text(s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def calculate_f1(predicted: str, truth: str) -> float:
        pred_tokens = Evaluator.normalize_text(predicted).split()
        truth_tokens = Evaluator.normalize_text(truth).split()

        # If either is empty
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
        num_same = sum(common_tokens.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def calculate_metrics(generated_answer: str, ground_truth: str) -> dict:
        """
        Calculates F1 Score and Token Overlap.
        """
        f1_score = Evaluator.calculate_f1(generated_answer, ground_truth)
        
        # Also keep a simpler containment check for debugging
        gen_norm = Evaluator.normalize_text(generated_answer)
        truth_norm = Evaluator.normalize_text(ground_truth)
        containment = 1.0 if truth_norm in gen_norm else 0.0
        
        return {
            "f1_score": f1_score,         # Use this as the main metric
            "containment": containment,
            "exact_match": 1.0 if gen_norm == truth_norm else 0.0
        }