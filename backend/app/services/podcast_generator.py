"""
Podcast Generator - Creates AI-generated podcast discussions from documents.
Uses local TTS (piper-tts) for fully offline audio generation.
"""
import asyncio
import os
import time
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import json

from app.core.config import settings
from app.services.llm_manager import LLMManager, GenerationConfig


@dataclass
class PodcastSegment:
    """A segment of the podcast (one speaker turn)."""
    speaker: str
    text: str
    voice: str  # 'male' or 'female'


@dataclass
class PodcastConfig:
    """Configuration for podcast generation."""
    duration_minutes: int = 10
    host_a_name: str = "Alex"
    host_b_name: str = "Sam"
    host_a_voice: str = "male"
    host_b_voice: str = "female"
    include_intro: bool = True
    include_music: bool = False
    speaking_rate: float = 1.0


class LocalTTS:
    """
    Offline TTS using piper-tts.
    Supports multiple voices for podcast generation.
    """

    def __init__(self):
        self.models_dir = settings.models_dir / "tts"
        self.male_model = self.models_dir / "en_US-lessac-medium.onnx"
        self.female_model = self.models_dir / "en_US-amy-medium.onnx"
        self._piper_available = self._check_piper()

    def _check_piper(self) -> bool:
        """Check if piper is available."""
        try:
            result = subprocess.run(
                ["piper", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def is_available(self) -> bool:
        """Check if TTS is available."""
        if not self._piper_available:
            return False
        return self.male_model.exists() and self.female_model.exists()

    async def synthesize(
        self,
        text: str,
        voice: str = "male",
        speed: float = 1.0,
    ) -> Optional[Path]:
        """
        Synthesize speech from text.

        Args:
            text: Text to speak
            voice: 'male' or 'female'
            speed: Speaking rate (0.5 to 2.0)

        Returns:
            Path to generated WAV file
        """
        if not self.is_available():
            return None

        # Select voice model
        model = self.male_model if voice == "male" else self.female_model
        config_file = str(model) + ".json"

        # Create temp output file
        output_file = settings.cache_dir / f"tts_{voice}_{int(time.time() * 1000)}.wav"

        # Build piper command
        cmd = [
            "piper",
            "--model", str(model),
            "--config", config_file,
            "--output_file", str(output_file),
            "--length_scale", str(1.0 / speed),
        ]

        # Run piper
        loop = asyncio.get_event_loop()

        def run_piper():
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate(input=text.encode('utf-8'))
            return process.returncode, stderr

        returncode, stderr = await loop.run_in_executor(None, run_piper)

        if returncode != 0:
            print(f"TTS failed: {stderr.decode()}")
            return None

        return output_file


class PodcastGenerator:
    """
    Generates podcast-style audio discussions from documents.

    Process:
    1. Analyze document and extract key topics
    2. Generate conversational script between two hosts
    3. Convert script to audio using TTS
    4. Combine audio segments into final podcast
    """

    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager
        self.tts = LocalTTS()
        self.outputs_dir = settings.outputs_dir / "podcasts"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        """Check if podcast generation is available."""
        return self.tts.is_available()

    async def generate_podcast(
        self,
        document_content: str,
        document_title: str,
        config: Optional[PodcastConfig] = None,
        progress_callback: Optional[callable] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a podcast from document content.

        Yields progress updates during generation.
        """
        config = config or PodcastConfig()

        # Step 1: Generate outline (10%)
        yield {"status": "generating", "progress": 5, "message": "Analyzing document..."}
        outline = await self._generate_outline(document_content, document_title)

        yield {"status": "generating", "progress": 15, "message": "Creating podcast outline..."}

        # Step 2: Generate script (30%)
        yield {"status": "generating", "progress": 20, "message": "Writing conversation script..."}
        segments = await self._generate_script(outline, config)

        yield {"status": "generating", "progress": 40, "message": f"Script ready: {len(segments)} segments"}

        # Step 3: Generate audio for each segment (60%)
        if not self.tts.is_available():
            # Return script only if TTS not available
            yield {
                "status": "complete",
                "progress": 100,
                "message": "TTS not available - returning script only",
                "script": [{"speaker": s.speaker, "text": s.text} for s in segments],
                "audio_url": None,
            }
            return

        yield {"status": "generating", "progress": 45, "message": "Generating audio..."}

        audio_files = []
        total_segments = len(segments)

        for i, segment in enumerate(segments):
            progress = 45 + int((i / total_segments) * 45)
            yield {
                "status": "generating",
                "progress": progress,
                "message": f"Recording segment {i + 1}/{total_segments}...",
            }

            audio_file = await self.tts.synthesize(
                segment.text,
                voice=segment.voice,
                speed=config.speaking_rate,
            )

            if audio_file:
                audio_files.append(audio_file)

        # Step 4: Combine audio (10%)
        yield {"status": "generating", "progress": 92, "message": "Combining audio..."}

        output_path = await self._combine_audio(
            audio_files,
            document_title,
            config,
        )

        # Clean up temp files
        for audio_file in audio_files:
            audio_file.unlink(missing_ok=True)

        yield {
            "status": "complete",
            "progress": 100,
            "message": "Podcast generated successfully!",
            "audio_url": f"/outputs/podcasts/{output_path.name}",
            "script": [{"speaker": s.speaker, "text": s.text} for s in segments],
            "duration_estimate": config.duration_minutes,
        }

    async def _generate_outline(
        self,
        content: str,
        title: str,
    ) -> Dict[str, Any]:
        """Generate podcast outline from document content."""
        prompt = f"""Analyze this document and create a podcast discussion outline.

Document Title: {title}

Document Content:
{content[:8000]}

Create an engaging outline for a podcast discussion with:
- 3-5 main topics to discuss
- Key insights and "aha moments"
- Interesting questions to explore
- Real-world implications
- Any counterintuitive findings

Output as JSON:
{{
    "title": "Podcast episode title",
    "topics": [
        {{
            "title": "Topic name",
            "key_points": ["point 1", "point 2"],
            "discussion_angle": "How hosts should approach this",
            "interesting_question": "A thought-provoking question"
        }}
    ],
    "key_takeaways": ["takeaway 1", "takeaway 2"],
    "hook": "Attention-grabbing opening statement"
}}"""

        response = await self.llm.generate_json(prompt)
        return response

    async def _generate_script(
        self,
        outline: Dict[str, Any],
        config: PodcastConfig,
    ) -> List[PodcastSegment]:
        """Generate conversational script from outline."""
        # Calculate target words (approx 150 words per minute)
        target_words = config.duration_minutes * 150

        prompt = f"""Create a {config.duration_minutes}-minute podcast script based on this outline:

{json.dumps(outline, indent=2)}

Characters:
- {config.host_a_name} (Host A): Enthusiastic expert who explains concepts clearly with analogies
- {config.host_b_name} (Host B): Curious learner who asks great questions and shares insights

Style Guidelines:
- Conversational, not formal or scripted-sounding
- Use analogies and real-world examples
- Natural back-and-forth with occasional humor
- Build excitement about insights
- Include "aha" moments and reactions

Format each line as:
{config.host_a_name}: [dialogue]
{config.host_b_name}: [dialogue]

Target length: ~{target_words} words

{f"Start with {config.host_a_name} welcoming listeners to the show." if config.include_intro else "Start directly with the first topic."}

Generate the full script:"""

        response = await self.llm.generate(
            prompt,
            GenerationConfig(max_tokens=target_words * 2, temperature=0.8)
        )

        # Parse script into segments
        segments = self._parse_script(response, config)
        return segments

    def _parse_script(
        self,
        script_text: str,
        config: PodcastConfig,
    ) -> List[PodcastSegment]:
        """Parse script text into speaker segments."""
        segments = []
        lines = script_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith(f"{config.host_a_name}:"):
                text = line[len(config.host_a_name) + 1:].strip()
                if text:
                    segments.append(PodcastSegment(
                        speaker=config.host_a_name,
                        text=text,
                        voice=config.host_a_voice,
                    ))
            elif line.startswith(f"{config.host_b_name}:"):
                text = line[len(config.host_b_name) + 1:].strip()
                if text:
                    segments.append(PodcastSegment(
                        speaker=config.host_b_name,
                        text=text,
                        voice=config.host_b_voice,
                    ))

        return segments

    async def _combine_audio(
        self,
        audio_files: List[Path],
        title: str,
        config: PodcastConfig,
    ) -> Path:
        """Combine audio segments into final podcast."""
        try:
            from pydub import AudioSegment
            from pydub.effects import normalize

            # Start with silence
            combined = AudioSegment.silent(duration=500)  # 0.5s intro

            for audio_file in audio_files:
                if audio_file.exists():
                    segment = AudioSegment.from_wav(str(audio_file))
                    combined += segment
                    combined += AudioSegment.silent(duration=300)  # 0.3s pause

            # Add outro silence
            combined += AudioSegment.silent(duration=500)

            # Normalize volume
            combined = normalize(combined)

            # Generate output filename
            safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:50]
            output_path = self.outputs_dir / f"{safe_title}_{int(time.time())}.mp3"

            # Export as MP3
            combined.export(
                str(output_path),
                format="mp3",
                bitrate="128k",
                tags={
                    "title": title,
                    "artist": "AI Podcast Generator",
                    "album": "Document DeepDive",
                },
            )

            return output_path

        except ImportError:
            # Fallback: just return first audio file if pydub not available
            if audio_files:
                output_path = self.outputs_dir / f"podcast_{int(time.time())}.wav"
                audio_files[0].rename(output_path)
                return output_path
            raise RuntimeError("pydub not installed and no audio files generated")

    async def generate_script_only(
        self,
        document_content: str,
        document_title: str,
        config: Optional[PodcastConfig] = None,
    ) -> Dict[str, Any]:
        """Generate just the podcast script without audio."""
        config = config or PodcastConfig()

        outline = await self._generate_outline(document_content, document_title)
        segments = await self._generate_script(outline, config)

        return {
            "title": outline.get("title", document_title),
            "outline": outline,
            "script": [
                {"speaker": s.speaker, "text": s.text, "voice": s.voice}
                for s in segments
            ],
            "estimated_duration_minutes": config.duration_minutes,
            "word_count": sum(len(s.text.split()) for s in segments),
        }