'''
Docstring for backend.app.services.setup_manager
Handles first time setup, Ollama installation and model downloads.
'''
import os
import sys
import json
import platform
import subprocess
import hashlib
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator, List, Callable
from dataclasses import dataclass, asdict
import psutil
import aiohttp
import aiofiles

from app.core.config import settings
from app.services.model_catalogue import (
    ModelCatalogue,
    MofrlInfo,
    DEFAULT_MODELS,
    get_model_catalogue,
)

@dataclass
class SystemInfo:
    # System info...
    platform:str
    platform_version:str
    cpu_count:int
    ram_gb:float
    disk_free_gb:float
    gpu_availabel:bool
    gpu_name: Optional[str]=None
    sufficient:bool=False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
