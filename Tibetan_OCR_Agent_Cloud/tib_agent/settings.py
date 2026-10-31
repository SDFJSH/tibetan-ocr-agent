"""Cloud settings: no hardcoded server paths, secrets are environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def flag(name, default=False):
    return os.environ.get(name, '1' if default else '0').strip().lower() in ('1', 'yes', 'true', 'on')

@dataclass(frozen=True)
class Settings:
    ocr_command: str = ''
    ocr_weights: str = str(ROOT / 'models' / 'best.pt')
    ocr_timeout_sec: int = 120
    ocr_confidence_calibrated: bool = False
    demo_mock_ocr: bool = False
    enable_qwen: bool = flag('ENABLE_QWEN')
    qwen_api_url: str = os.environ.get('QWEN_API_URL', '').strip()
    qwen_api_key: str = os.environ.get('QWEN_API_KEY', '').strip()
    qwen_model: str = os.environ.get('QWEN_MODEL', 'qwen3.5-flash').strip()
    qwen_timeout_sec: int = 22
    sharpness_threshold: float = 65.0
    contrast_threshold: float = 35.0
    max_image_mb: int = 8
    output_dir: Path = ROOT / 'outputs'

    def ocr_ready(self):
        return Path(self.ocr_weights).is_file()
