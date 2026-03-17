"""
Accuracy Evaluation Module for Thesis Testing

This module provides:
1. Character-level accuracy comparing detected text against ground-truth
2. Translation accuracy using the BLEU (Bilingual Evaluation Understudy) algorithm

Reference:
- Papineni, K., et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation
"""

import re
import json
import unicodedata
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass
import math


@dataclass
class CharacterAccuracyResult:
    """Container for character-level accuracy results"""
    ground_truth: str
    detected_text: str
    correct_chars: int
    total_chars: int
    accuracy: float
    insertions: int
    deletions: int
    substitutions: int
    character_error_rate: float  # CER


@dataclass
class BLEUResult:
    """Container for BLEU score results"""
    reference: str
    hypothesis: str
    bleu_score: float
    precisions: List[float]  # 1-gram to 4-gram precisions
    brevity_penalty: float
    reference_length: int
    hypothesis_length: int


class CharacterAccuracy:
    """
    Computes character-level accuracy between detected text and ground truth.
    Uses Levenshtein distance for precise character-by-character comparison.
    """
    
    @staticmethod
    def normalize_text(text: str, ignore_whitespace: bool = True, 
                       ignore_punctuation: bool = False,
                       lowercase: bool = False) -> str:
        """
        Normalize text for fair comparison.
        
        Args:
            text: Input text
            ignore_whitespace: Remove all whitespace
            ignore_punctuation: Remove punctuation marks
            lowercase: Convert to lowercase
            
        Returns:
            Normalized text string
        """
        if text is None:
            return ""
        
        # Unicode normalization (NFKC for CJK compatibility)
        text = unicodedata.normalize('NFKC', text)
        
        if lowercase:
            text = text.lower()
        
        if ignore_whitespace:
            text = re.sub(r'\s+', '', text)
        
        if ignore_punctuation:
            # Remove common punctuation (CJK and Western)
            text = re.sub(r'[。、！？「」『』【】（）《》〈〉・…\.,!?"\'()[\]{}:;]', '', text)
        
        return text
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> Tuple[int, int, int, int]:
        """
        Calculate Levenshtein distance with edit operation breakdown.
        
        Returns:
            Tuple of (distance, insertions, deletions, substitutions)
        """
        if len(s1) < len(s2):
            return CharacterAccuracy.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1), 0, len(s1), 0
        
        # Previous row distances
        previous_row = list(range(len(s2) + 1))
        
        # Track operations
        insertions = 0
        deletions = 0
        substitutions = 0
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Costs for operations
                insert_cost = previous_row[j + 1] + 1
                delete_cost = current_row[j] + 1
                substitute_cost = previous_row[j] + (0 if c1 == c2 else 1)
                
                min_cost = min(insert_cost, delete_cost, substitute_cost)
                current_row.append(min_cost)
            
            previous_row = current_row
        
        # Calculate operations using traceback
        distance = previous_row[-1]
        
        # Approximate operation breakdown
        len_diff = len(s1) - len(s2)
        if len_diff > 0:
            deletions = len_diff
        else:
            insertions = -len_diff
        
        # Remaining distance is substitutions
        substitutions = max(0, distance - abs(len_diff))
        
        return distance, insertions, deletions, substitutions
    
    @classmethod
    def compute_accuracy(cls, ground_truth: str, detected_text: str,
                        normalize: bool = True,
                        ignore_whitespace: bool = True,
                        ignore_punctuation: bool = False) -> CharacterAccuracyResult:
        """
        Compute character-level accuracy between ground truth and detected text.
        
        Args:
            ground_truth: Reference text (correct transcription)
            detected_text: OCR-detected text
            normalize: Whether to normalize text before comparison
            ignore_whitespace: Ignore whitespace differences
            ignore_punctuation: Ignore punctuation differences
            
        Returns:
            CharacterAccuracyResult with detailed metrics
        """
        if normalize:
            gt_normalized = cls.normalize_text(ground_truth, ignore_whitespace, ignore_punctuation)
            det_normalized = cls.normalize_text(detected_text, ignore_whitespace, ignore_punctuation)
        else:
            gt_normalized = ground_truth or ""
            det_normalized = detected_text or ""
        
        total_chars = len(gt_normalized)
        
        if total_chars == 0:
            return CharacterAccuracyResult(
                ground_truth=ground_truth or "",
                detected_text=detected_text or "",
                correct_chars=0,
                total_chars=0,
                accuracy=1.0 if len(det_normalized) == 0 else 0.0,
                insertions=len(det_normalized),
                deletions=0,
                substitutions=0,
                character_error_rate=0.0 if len(det_normalized) == 0 else float('inf')
            )
        
        distance, insertions, deletions, substitutions = cls.levenshtein_distance(
            gt_normalized, det_normalized
        )
        
        # Correct characters = total - errors
        correct_chars = max(0, total_chars - distance)
        accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        
        # Character Error Rate (CER)
        cer = distance / total_chars if total_chars > 0 else 0.0
        
        return CharacterAccuracyResult(
            ground_truth=ground_truth or "",
            detected_text=detected_text or "",
            correct_chars=correct_chars,
            total_chars=total_chars,
            accuracy=accuracy,
            insertions=insertions,
            deletions=deletions,
            substitutions=substitutions,
            character_error_rate=cer
        )


class BLEUScore:
    """
    Computes BLEU (Bilingual Evaluation Understudy) score for translation evaluation.
    
    BLEU measures translation quality by comparing n-gram overlaps between
    the machine translation (hypothesis) and reference translation.
    
    Reference:
        Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002).
        BLEU: a method for automatic evaluation of machine translation.
    """
    
    @staticmethod
    def tokenize(text: str, language: str = 'auto') -> List[str]:
        """
        Tokenize text into words/characters based on language.
        
        For CJK languages, use character-level tokenization.
        For Western languages, use word-level tokenization.
        """
        if text is None:
            return []
        
        text = text.strip()
        
        # Detect CJK content
        has_cjk = any('\u4e00' <= char <= '\u9fff' or 
                      '\u3040' <= char <= '\u309f' or
                      '\u30a0' <= char <= '\u30ff' or
                      '\uac00' <= char <= '\ud7af'
                      for char in text)
        
        if has_cjk or language in ['JPN', 'CHS', 'CHT', 'KOR']:
            # Character-level tokenization for CJK
            # But keep Western words together
            tokens = []
            current_word = ""
            for char in text:
                if char.isspace():
                    if current_word:
                        tokens.append(current_word)
                        current_word = ""
                elif '\u4e00' <= char <= '\u9fff' or \
                     '\u3040' <= char <= '\u309f' or \
                     '\u30a0' <= char <= '\u30ff' or \
                     '\uac00' <= char <= '\ud7af':
                    if current_word:
                        tokens.append(current_word)
                        current_word = ""
                    tokens.append(char)
                else:
                    current_word += char
            if current_word:
                tokens.append(current_word)
            return tokens
        else:
            # Word-level tokenization for Western languages
            return text.lower().split()
    
    @staticmethod
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        """Get n-grams from token list"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams
    
    @classmethod
    def compute_bleu(cls, reference: str, hypothesis: str,
                    max_n: int = 4,
                    weights: Optional[List[float]] = None,
                    smoothing: bool = True,
                    language: str = 'auto') -> BLEUResult:
        """
        Compute BLEU score.
        
        Args:
            reference: Reference (correct) translation
            hypothesis: Machine translation output
            max_n: Maximum n-gram order (default: 4)
            weights: Weights for each n-gram precision (default: uniform)
            smoothing: Apply smoothing for zero counts
            language: Target language for tokenization
            
        Returns:
            BLEUResult with detailed metrics
        """
        if weights is None:
            weights = [1.0 / max_n] * max_n
        
        ref_tokens = cls.tokenize(reference, language)
        hyp_tokens = cls.tokenize(hypothesis, language)
        
        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)
        
        if hyp_len == 0:
            return BLEUResult(
                reference=reference or "",
                hypothesis=hypothesis or "",
                bleu_score=0.0,
                precisions=[0.0] * max_n,
                brevity_penalty=0.0,
                reference_length=ref_len,
                hypothesis_length=hyp_len
            )
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = cls.get_ngrams(ref_tokens, n)
            hyp_ngrams = cls.get_ngrams(hyp_tokens, n)
            
            # Count clipped matches
            matched = 0
            total = 0
            for ngram, count in hyp_ngrams.items():
                matched += min(count, ref_ngrams.get(ngram, 0))
                total += count
            
            if total == 0:
                precision = 0.0
            else:
                precision = matched / total
            
            # Apply smoothing for zero precisions (except unigram)
            if smoothing and precision == 0 and n > 1:
                denominator = max(1, hyp_len - n + 2)
                precision = 1.0 / denominator
            
            precisions.append(precision)
        
        # Calculate brevity penalty
        if hyp_len >= ref_len:
            bp = 1.0
        elif ref_len == 0:
            bp = 0.0
        else:
            bp = math.exp(1 - ref_len / hyp_len)
        
        # Calculate weighted geometric mean of precisions
        if all(p > 0 for p in precisions):
            log_precisions = [w * math.log(p) if p > 0 else float('-inf') 
                            for w, p in zip(weights, precisions)]
            bleu = bp * math.exp(sum(log_precisions))
        else:
            bleu = 0.0
        
        return BLEUResult(
            reference=reference or "",
            hypothesis=hypothesis or "",
            bleu_score=bleu,
            precisions=precisions,
            brevity_penalty=bp,
            reference_length=ref_len,
            hypothesis_length=hyp_len
        )


class AccuracyEvaluator:
    """
    High-level evaluator for comprehensive accuracy testing.
    Combines OCR accuracy and translation BLEU scoring.
    """
    
    def __init__(self):
        self.char_accuracy = CharacterAccuracy()
        self.bleu_scorer = BLEUScore()
        self.results = {
            'ocr': [],
            'translation': []
        }
    
    def evaluate_ocr(self, ground_truth: str, detected_text: str,
                    image_id: str = None) -> CharacterAccuracyResult:
        """
        Evaluate OCR accuracy for a single text region.
        
        Args:
            ground_truth: Reference text (human transcription)
            detected_text: OCR output
            image_id: Optional identifier for the image
            
        Returns:
            CharacterAccuracyResult
        """
        result = self.char_accuracy.compute_accuracy(ground_truth, detected_text)
        self.results['ocr'].append({
            'image_id': image_id,
            'result': result
        })
        return result
    
    def evaluate_translation(self, reference: str, hypothesis: str,
                            image_id: str = None,
                            target_lang: str = 'ENG') -> BLEUResult:
        """
        Evaluate translation accuracy using BLEU.
        
        Args:
            reference: Reference (human) translation
            hypothesis: Machine translation output
            image_id: Optional identifier
            target_lang: Target language code
            
        Returns:
            BLEUResult
        """
        result = self.bleu_scorer.compute_bleu(reference, hypothesis, language=target_lang)
        self.results['translation'].append({
            'image_id': image_id,
            'result': result
        })
        return result
    
    def get_aggregate_ocr_accuracy(self) -> Dict:
        """Calculate aggregate OCR accuracy metrics"""
        if not self.results['ocr']:
            return {'mean_accuracy': 0.0, 'mean_cer': 0.0, 'sample_count': 0}
        
        accuracies = [r['result'].accuracy for r in self.results['ocr']]
        cers = [r['result'].character_error_rate for r in self.results['ocr']]
        
        return {
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'mean_cer': sum(cers) / len(cers),
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies),
            'sample_count': len(accuracies)
        }
    
    def get_aggregate_bleu(self) -> Dict:
        """Calculate aggregate BLEU scores"""
        if not self.results['translation']:
            return {'mean_bleu': 0.0, 'sample_count': 0}
        
        bleu_scores = [r['result'].bleu_score for r in self.results['translation']]
        
        return {
            'mean_bleu': sum(bleu_scores) / len(bleu_scores),
            'min_bleu': min(bleu_scores),
            'max_bleu': max(bleu_scores),
            'sample_count': len(bleu_scores)
        }
    
    def export_results(self, filepath: str = None) -> Dict:
        """
        Export all evaluation results.
        
        Args:
            filepath: Optional path to save JSON results
            
        Returns:
            Dictionary with all results
        """
        export_data = {
            'ocr_results': [
                {
                    'image_id': r['image_id'],
                    'ground_truth': r['result'].ground_truth,
                    'detected_text': r['result'].detected_text,
                    'accuracy': r['result'].accuracy,
                    'cer': r['result'].character_error_rate,
                    'correct_chars': r['result'].correct_chars,
                    'total_chars': r['result'].total_chars
                }
                for r in self.results['ocr']
            ],
            'translation_results': [
                {
                    'image_id': r['image_id'],
                    'reference': r['result'].reference,
                    'hypothesis': r['result'].hypothesis,
                    'bleu_score': r['result'].bleu_score,
                    'precisions': r['result'].precisions,
                    'brevity_penalty': r['result'].brevity_penalty
                }
                for r in self.results['translation']
            ],
            'aggregate': {
                'ocr': self.get_aggregate_ocr_accuracy(),
                'translation': self.get_aggregate_bleu()
            }
        }
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return export_data
    
    def reset(self):
        """Clear all stored results"""
        self.results = {'ocr': [], 'translation': []}


# Convenience function for quick BLEU calculation
def calculate_bleu(reference: str, hypothesis: str, language: str = 'auto') -> float:
    """
    Quick function to calculate BLEU score.
    
    Args:
        reference: Reference translation
        hypothesis: Machine translation
        language: Target language
        
    Returns:
        BLEU score (0.0 to 1.0)
    """
    return BLEUScore.compute_bleu(reference, hypothesis, language=language).bleu_score


# Convenience function for quick character accuracy
def calculate_character_accuracy(ground_truth: str, detected: str) -> float:
    """
    Quick function to calculate character accuracy.
    
    Args:
        ground_truth: Reference text
        detected: OCR output
        
    Returns:
        Accuracy (0.0 to 1.0)
    """
    return CharacterAccuracy.compute_accuracy(ground_truth, detected).accuracy
