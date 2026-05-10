from typing import Optional, Any

try:
    from pocket_tts import TTSModel
    import torch
    POCKET_TTS_AVAILABLE = True
except ImportError:
    POCKET_TTS_AVAILABLE = False
    TTSModel = None
    torch = None

from app.core.logging import logger

tts_model: Optional[Any] = None
voice_states: dict[str, Any] = {}


AVAILABLE_VOICES = [
    {"name": "alba", "language": "en"},
    {"name": "giovanni", "language": "it"},
    {"name": "lola", "language": "es"},
    {"name": "juergen", "language": "de"},
    {"name": "rafael", "language": "pt"},
    {"name": "estelle", "language": "fr"},
    {"name": "anna", "language": "en"},
    {"name": "azelma", "language": "en"},
    {"name": "bill_boerst", "language": "en"},
    {"name": "caro_davy", "language": "en"},
    {"name": "charles", "language": "en"},
    {"name": "cosette", "language": "en"},
    {"name": "eponine", "language": "en"},
    {"name": "eve", "language": "en"},
    {"name": "fantine", "language": "en"},
    {"name": "george", "language": "en"},
    {"name": "jane", "language": "en"},
    {"name": "jean", "language": "en"},
    {"name": "javert", "language": "en"},
    {"name": "marius", "language": "en"},
    {"name": "mary", "language": "en"},
    {"name": "michael", "language": "en"},
    {"name": "paul", "language": "en"},
    {"name": "peter_yearsley", "language": "en"},
    {"name": "stuart_bell", "language": "en"},
    {"name": "vera", "language": "en"},
]


def load_tts_model() -> None:
    global tts_model, voice_states
    if not POCKET_TTS_AVAILABLE:
        logger.warning("pocket-tts not installed. Run: pip install pocket-tts scipy")
        return
    
    try:
        logger.info("Loading TTS model...")
        tts_model = TTSModel.load_model()
        
        for voice in AVAILABLE_VOICES:
            try:
                state = tts_model.get_state_for_audio_prompt(voice["name"])
                voice_states[voice["name"]] = state
                logger.info(f"Loaded voice: {voice['name']}")
            except Exception as e:
                logger.warning(f"Failed to load voice {voice['name']}: {e}")
        
        logger.info(f"TTS model loaded successfully. Sample rate: {tts_model.sample_rate}")
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        raise


def get_tts_model() -> Optional[Any]:
    return tts_model


def get_voice_state(voice_name: str):
    return voice_states.get(voice_name)


def get_sample_rate() -> int:
    if tts_model is not None:
        return tts_model.sample_rate
    return 24000


def is_tts_loaded() -> bool:
    return tts_model is not None