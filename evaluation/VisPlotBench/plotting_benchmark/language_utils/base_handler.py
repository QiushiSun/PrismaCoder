from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from pathlib import Path

class LanguageHandler(ABC):
    """Base class for language-specific handlers"""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config
    
    @abstractmethod
    def extract_plotting_code(self, response: str) -> str:
        """Extract plotting code from model response"""
        pass
    
    @abstractmethod
    def build_plots(self, dataset: pd.DataFrame, output_path: Path, csv_folder: Path) -> None:
        """
        Build plots for the entire dataset.
        """
        pass
    
    @abstractmethod
    def parse_plots_notebook(self, plots_path: Path) -> pd.DataFrame:
        """
        Parse the generated notebook and extract results.
        """
        pass