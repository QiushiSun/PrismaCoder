from typing import Dict, Any
from .base_handler import LanguageHandler
from .python_utils import PythonHandler
from .vegalite_utils import VegaLiteHandler
from .mermaid_utils import MermaidHandler
from .lilypond_utils import LilyPondHandler
from .svg_utils import SVGHandler
from .asymptote_utils import AsymptoteHandler
from .latex_utils import LaTexHandler
from .html_utils import HTMLHandler

def get_language_handler(language: str, config: Dict[str, Any] = None) -> LanguageHandler:
    language = language.lower()
    
    handlers = {
        "python": PythonHandler,
        "vegalite": VegaLiteHandler,
        "mermaid": MermaidHandler,
        "lilypond": LilyPondHandler,
        "svg": SVGHandler,
        "asymptote": AsymptoteHandler,
        "latex": LaTexHandler,
        "html": HTMLHandler,
    }
    
    handler_class = handlers.get(language)
    if handler_class is None:
        raise ValueError(f"Unsupported language: {language}")
    
    return handler_class(config=config)

__all__ = ['LanguageHandler', 'get_language_handler']