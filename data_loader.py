"""
Data loading module for scientific claim verification datasets.

Supports: SciFact, Climate-FEVER, FEVER
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ClaimVerificationDataset(Dataset):
    """
    PyTorch Dataset for claim verification tasks.
    
    Supports multiple datasets:
    - SciFact: 1,409 claims with biomedical abstracts
    - Climate-FEVER: 1,535 claims with climate documents
    - FEVER: 19,998 claims with Wikipedia passages
    
    Example:
        >>> dataset = ClaimVerificationDataset(dataset_name='scifact', split='train')
        >>> sample = dataset[0]
        >>> print(sample['claim'])
        >>> print(sample['evidence'])
    """
    
    def __init__(
        self, 
        dataset_name: str = 'scifact',
        split: str = 'train',
        max_evidence: int = 5
    ):
        """
        Initialize dataset.
        
        Args:
            dataset_name: 'scifact', 'climatefever', or 'fever'
            split: 'train', 'validation', or 'test'
            max_evidence: Maximum number of evidence documents per claim
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_evidence = max_evidence
        
        # Load dataset from HuggingFace
        logger.info(f"Loading {dataset_name} dataset ({split} split)...")
        
        try:
            if dataset_name.lower() == 'scifact':
                self.data = load_dataset('scifact', split=split)
            elif dataset_name.lower() == 'climatefever':
                self.data = load_dataset('climate_fever', split=split)
            elif dataset_name.lower() == 'fever':
                self.data = load_dataset('fever', split=split)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
                
            logger.info(f"Loaded {len(self.data)} samples")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with keys:
            - claim: The claim text
            - evidence: List of evidence passages
            - label: Verdict (SUPPORTED/CONTRADICTED/NOINFO)
            - evidence_ids: IDs of evidence documents
        """
        sample = self.data[idx]
        
        # Standardize across datasets
        item = {
            'claim': sample.get('claim', sample.get('text', '')),
            'evidence': self._extract_evidence(sample),
            'label': self._extract_label(sample),
            'evidence_ids': self._extract_evidence_ids(sample),
            'dataset': self.dataset_name
        }
        
        return item
    
    def _extract_evidence(self, sample: Dict) -> List[str]:
        """Extract evidence passages from sample."""
        evidence = []
        
        if 'evidence' in sample:
            if isinstance(sample['evidence'], list):
                for ev in sample['evidence'][:self.max_evidence]:
                    if isinstance(ev, dict) and 'text' in ev:
                        evidence.append(ev['text'])
                    elif isinstance(ev, str):
                        evidence.append(ev)
        
        elif 'evidence_text' in sample:
            evidence = [sample['evidence_text']]
        
        elif 'abstract' in sample:
            evidence = [sample['abstract']]
        
        return evidence[:self.max_evidence]
    
    def _extract_label(self, sample: Dict) -> str:
        """Extract verdict label from sample."""
        if 'label' in sample:
            label = sample['label']
            if isinstance(label, int):
                # Convert numeric labels to strings
                label_map = {0: 'NOINFO', 1: 'SUPPORTED', 2: 'CONTRADICTED'}
                return label_map.get(label, 'NOINFO')
            return str(label)
        return 'NOINFO'
    
    def _extract_evidence_ids(self, sample: Dict) -> List[int]:
        """Extract evidence document IDs."""
        ids = []
        if 'evidence_ids' in sample:
            ids = sample['evidence_ids']
        elif 'doc_id' in sample:
            ids = [sample['doc_id']]
        return ids


class ClaimVerificationDataLoader:
    """Utility for loading and batching claim verification data."""
    
    @staticmethod
    def get_dataset(
        dataset_name: str = 'scifact',
        split: str = 'train',
        max_evidence: int = 5
    ) -> ClaimVerificationDataset:
        """
        Load a claim verification dataset.
        
        Args:
            dataset_name: 'scifact', 'climatefever', or 'fever'
            split: 'train', 'validation', or 'test'
            max_evidence: Maximum evidence passages per claim
            
        Returns:
            ClaimVerificationDataset instance
        """
        return ClaimVerificationDataset(
            dataset_name=dataset_name,
            split=split,
            max_evidence=max_evidence
        )
    
    @staticmethod
    def get_train_dev_split(
        dataset_name: str = 'scifact',
        max_evidence: int = 5
    ) -> Tuple[ClaimVerificationDataset, ClaimVerificationDataset]:
        """
        Get train and validation datasets.
        
        Args:
            dataset_name: Dataset to load
            max_evidence: Maximum evidence passages
            
        Returns:
            Tuple of (train_dataset, dev_dataset)
        """
        train_set = ClaimVerificationDataset(
            dataset_name=dataset_name,
            split='train',
            max_evidence=max_evidence
        )
        
        dev_set = ClaimVerificationDataset(
            dataset_name=dataset_name,
            split='validation',
            max_evidence=max_evidence
        )
        
        return train_set, dev_set


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Load SciFact dataset
    dataset = ClaimVerificationDataset(dataset_name='scifact', split='train')
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample claim: {sample['claim']}")
    print(f"Number of evidence: {len(sample['evidence'])}")
    print(f"Label: {sample['label']}")
