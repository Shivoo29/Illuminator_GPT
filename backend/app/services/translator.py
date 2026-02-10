"""
Offline Translator - Local translation using Opus-MT models.
Supports multiple language pairs, fully offline after model download.
"""
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class LanguagePair:
    """Language pair configuration."""
    source: str
    target: str
    model_name: str
    display_name: str
    size_mb: int = 300


# Available language pairs
LANGUAGE_PAIRS: List[LanguagePair] = [
    LanguagePair("en", "hi", "opus-mt-en-hi", "English → Hindi", 300),
    LanguagePair("hi", "en", "opus-mt-hi-en", "Hindi → English", 300),
    LanguagePair("en", "es", "opus-mt-en-es", "English → Spanish", 300),
    LanguagePair("es", "en", "opus-mt-es-en", "Spanish → English", 300),
    LanguagePair("en", "fr", "opus-mt-en-fr", "English → French", 300),
    LanguagePair("fr", "en", "opus-mt-fr-en", "French → English", 300),
    LanguagePair("en", "de", "opus-mt-en-de", "English → German", 300),
    LanguagePair("de", "en", "opus-mt-de-en", "German → English", 300),
    LanguagePair("en", "zh", "opus-mt-en-zh", "English → Chinese", 300),
    LanguagePair("zh", "en", "opus-mt-zh-en", "Chinese → English", 300),
]


class OfflineTranslator:
    """
    Offline translation using Hugging Face Opus-MT models.

    Models are downloaded once and cached locally for offline use.
    Each language pair is ~300MB.
    """

    def __init__(self):
        self.models_dir = settings.models_dir / "translation"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Cache loaded models
        self._loaded_models: Dict[str, Any] = {}

    def get_available_pairs(self) -> List[Dict[str, Any]]:
        """Get list of available language pairs."""
        pairs = []
        for pair in LANGUAGE_PAIRS:
            installed = self._is_pair_installed(pair)
            pairs.append({
                "source": pair.source,
                "target": pair.target,
                "display_name": pair.display_name,
                "model_name": pair.model_name,
                "size_mb": pair.size_mb,
                "installed": installed,
            })
        return pairs

    def _is_pair_installed(self, pair: LanguagePair) -> bool:
        """Check if a language pair model is installed."""
        model_path = self.models_dir / pair.model_name
        return model_path.exists() and (model_path / "config.json").exists()

    def is_available(self, source: str, target: str) -> bool:
        """Check if translation is available for a language pair."""
        pair = self._find_pair(source, target)
        return pair is not None and self._is_pair_installed(pair)

    def _find_pair(self, source: str, target: str) -> Optional[LanguagePair]:
        """Find language pair configuration."""
        for pair in LANGUAGE_PAIRS:
            if pair.source == source and pair.target == target:
                return pair
        return None

    async def _load_model(self, source: str, target: str):
        """Load translation model for a language pair."""
        pair_key = f"{source}-{target}"

        if pair_key in self._loaded_models:
            return self._loaded_models[pair_key]

        pair = self._find_pair(source, target)
        if not pair:
            raise ValueError(f"Language pair not supported: {source} → {target}")

        if not self._is_pair_installed(pair):
            raise ValueError(f"Model not installed: {pair.display_name}")

        try:
            from transformers import MarianMTModel, MarianTokenizer

            model_path = self.models_dir / pair.model_name

            loop = asyncio.get_event_loop()

            def load():
                tokenizer = MarianTokenizer.from_pretrained(str(model_path))
                model = MarianMTModel.from_pretrained(str(model_path))
                return model, tokenizer

            model, tokenizer = await loop.run_in_executor(None, load)

            self._loaded_models[pair_key] = (model, tokenizer)
            return model, tokenizer

        except ImportError:
            raise RuntimeError("transformers library not installed")

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> Dict[str, Any]:
        """
        Translate text between languages.

        Args:
            text: Text to translate
            source_lang: Source language code (en, hi, es, etc.)
            target_lang: Target language code

        Returns:
            Dict with translated text and metadata
        """
        try:
            model, tokenizer = await self._load_model(source_lang, target_lang)

            loop = asyncio.get_event_loop()

            def do_translate():
                # Tokenize
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

                # Generate translation
                translated = model.generate(**inputs)

                # Decode
                result = tokenizer.decode(translated[0], skip_special_tokens=True)
                return result

            translated_text = await loop.run_in_executor(None, do_translate)

            return {
                "success": True,
                "original": text,
                "translated": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original": text,
            }

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> Dict[str, Any]:
        """Translate multiple texts."""
        try:
            model, tokenizer = await self._load_model(source_lang, target_lang)

            loop = asyncio.get_event_loop()

            def do_translate():
                results = []
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    translated = model.generate(**inputs)
                    result = tokenizer.decode(translated[0], skip_special_tokens=True)
                    results.append(result)
                return results

            translated_texts = await loop.run_in_executor(None, do_translate)

            return {
                "success": True,
                "translations": [
                    {"original": orig, "translated": trans}
                    for orig, trans in zip(texts, translated_texts)
                ],
                "source_lang": source_lang,
                "target_lang": target_lang,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of text.
        Uses a simple heuristic based on character analysis.
        For more accurate detection, a dedicated model would be needed.
        """
        # Simple character-based detection
        text_lower = text.lower()

        # Check for specific scripts
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)

        if has_devanagari:
            return {"detected_language": "hi", "confidence": 0.9, "script": "Devanagari"}
        elif has_chinese:
            return {"detected_language": "zh", "confidence": 0.9, "script": "Chinese"}
        elif has_arabic:
            return {"detected_language": "ar", "confidence": 0.9, "script": "Arabic"}

        # Check for common words in European languages
        spanish_words = ["el", "la", "de", "que", "en", "es", "por", "con"]
        french_words = ["le", "la", "de", "et", "en", "est", "que", "pour"]
        german_words = ["der", "die", "und", "in", "den", "ist", "von", "mit"]

        words = text_lower.split()

        spanish_count = sum(1 for w in words if w in spanish_words)
        french_count = sum(1 for w in words if w in french_words)
        german_count = sum(1 for w in words if w in german_words)

        if spanish_count > french_count and spanish_count > german_count and spanish_count > 1:
            return {"detected_language": "es", "confidence": 0.7, "method": "word_frequency"}
        elif french_count > spanish_count and french_count > german_count and french_count > 1:
            return {"detected_language": "fr", "confidence": 0.7, "method": "word_frequency"}
        elif german_count > spanish_count and german_count > french_count and german_count > 1:
            return {"detected_language": "de", "confidence": 0.7, "method": "word_frequency"}

        # Default to English
        return {"detected_language": "en", "confidence": 0.5, "method": "default"}

    def get_status(self) -> Dict[str, Any]:
        """Get translator status."""
        installed_pairs = [
            p for p in LANGUAGE_PAIRS if self._is_pair_installed(p)
        ]

        return {
            "available": len(installed_pairs) > 0,
            "installed_pairs": [
                {"source": p.source, "target": p.target, "display_name": p.display_name}
                for p in installed_pairs
            ],
            "total_pairs_available": len(LANGUAGE_PAIRS),
            "loaded_models": list(self._loaded_models.keys()),
        }