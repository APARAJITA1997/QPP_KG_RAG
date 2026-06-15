"""
Evidence-grounded answer generation module.

Uses Flan-T5 to generate answers based on retrieved passages.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class EvidenceGroundedGenerator:
    """
    Generate answers grounded in retrieved evidence.
    
    Uses Flan-T5 to read retrieved passages and write
    answers that are faithful to the evidence.
    
    Example:
        >>> generator = EvidenceGroundedGenerator()
        >>> passages = ["Insulin is a hormone produced by the pancreas."]
        >>> answer = generator.generate(
        ...     "What is insulin?",
        ...     passages
        ... )
        >>> print(answer)
        "Insulin is a hormone produced by the pancreas."
    """
    
    def __init__(
        self,
        model: str = 'google/flan-t5-base',
        device: str = 'cpu',
        max_length: int = 128
    ):
        """
        Initialize generator.
        
        Args:
            model: Flan-T5 model name
            device: 'cpu' or 'cuda'
            max_length: Maximum answer length
        """
        logger.info(f"Loading model: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
    
    def generate(
        self,
        question: str,
        passages: List[str],
        max_passages: int = 5
    ) -> str:
        """
        Generate answer for a question using passages.
        
        Args:
            question: Question text
            passages: List of evidence passages
            max_passages: Max passages to use
            
        Returns:
            Generated answer
        """
        # Limit passages
        passages = passages[:max_passages]
        
        if not passages:
            return "No evidence provided."
        
        # Construct prompt
        evidence_text = " ".join(passages)
        prompt = (
            f"Answer this question based only on the evidence below:\n"
            f"Question: {question}\n"
            f"Evidence: {evidence_text}\n"
            f"Answer:"
        )
        
        # Encode
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer.strip()
    
    def generate_batch(
        self,
        questions: List[str],
        passages_list: List[List[str]],
        max_passages: int = 5
    ) -> List[str]:
        """
        Generate answers for multiple questions.
        
        Args:
            questions: List of questions
            passages_list: List of passage lists
            max_passages: Max passages per question
            
        Returns:
            List of generated answers
        """
        answers = []
        for question, passages in zip(questions, passages_list):
            answer = self.generate(question, passages, max_passages)
            answers.append(answer)
        return answers
    
    def generate_with_info(
        self,
        question: str,
        passages: List[str],
        max_passages: int = 5
    ) -> Dict:
        """
        Generate answer with metadata.
        
        Args:
            question: Question text
            passages: Evidence passages
            max_passages: Max passages
            
        Returns:
            Dictionary with answer and metadata
        """
        answer = self.generate(question, passages, max_passages)
        
        info = {
            'question': question,
            'answer': answer,
            'num_passages': len(passages[:max_passages]),
            'passages_used': passages[:max_passages]
        }
        
        return info


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Initialize generator
    generator = EvidenceGroundedGenerator()
    
    # Example
    question = "What is insulin?"
    passages = [
        "Insulin is a hormone produced by the pancreas.",
        "It regulates blood glucose levels.",
        "Insulin is essential for glucose metabolism."
    ]
    
    # Generate
    answer = generator.generate(question, passages)
    print(f"Q: {question}")
    print(f"A: {answer}")
