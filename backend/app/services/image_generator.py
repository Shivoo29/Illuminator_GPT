"""
Offline Image Generator - Local Stable Diffusion using ONNX Runtime.
"""
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any

from app.core.config import settings


class OfflineImageGenerator:
    """
    Offline image generation using Stable Diffusion ONNX.
    Works on CPU or GPU, fully offline after model download.
    """

    def __init__(self):
        self.models_dir = settings.models_dir / "image_gen"
        self.outputs_dir = settings.outputs_dir / "images"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.pipe = None
        self._initialized = False

    def is_available(self) -> bool:
        """Check if image generation is available."""
        model_index = self.models_dir / "model_index.json"
        return model_index.exists()

    async def initialize(self) -> bool:
        """Initialize the Stable Diffusion pipeline."""
        if self._initialized:
            return True

        if not self.is_available():
            return False

        try:
            from diffusers import OnnxStableDiffusionPipeline

            loop = asyncio.get_event_loop()

            def load():
                pipe = OnnxStableDiffusionPipeline.from_pretrained(
                    str(self.models_dir),
                    provider="CPUExecutionProvider",
                )
                return pipe

            self.pipe = await loop.run_in_executor(None, load)
            self._initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize image generator: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
    ) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image
            negative_prompt: What to avoid in the image
            width: Image width (must be multiple of 8)
            height: Image height (must be multiple of 8)
            num_steps: Number of inference steps
            guidance_scale: How closely to follow the prompt

        Returns:
            Dict with image path and metadata
        """
        if not self._initialized:
            if not await self.initialize():
                return {
                    "success": False,
                    "error": "Image generation not available. Please install the model first.",
                }

        # Ensure dimensions are valid
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Default negative prompt
        if not negative_prompt:
            negative_prompt = "blurry, bad quality, distorted, ugly, deformed"

        loop = asyncio.get_event_loop()

        def generate():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            )
            return result.images[0]

        try:
            image = await loop.run_in_executor(None, generate)

            # Save image
            timestamp = int(time.time())
            filename = f"generated_{timestamp}.png"
            output_path = self.outputs_dir / filename
            image.save(str(output_path))

            return {
                "success": True,
                "image_url": f"/outputs/images/{filename}",
                "path": str(output_path),
                "prompt": prompt,
                "dimensions": f"{width}x{height}",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def get_status(self) -> Dict[str, Any]:
        """Get image generator status."""
        return {
            "available": self.is_available(),
            "initialized": self._initialized,
            "model_path": str(self.models_dir),
            "outputs_path": str(self.outputs_dir),
        }