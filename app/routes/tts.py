from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io
import scipy.io.wavfile
import torch

from app.schemas import TTSRequest, TTSVoicesResponse, TTSVoiceInfo
from app.core.tts import (
    load_tts_model,
    get_tts_model,
    get_voice_state,
    get_sample_rate,
    is_tts_loaded,
    AVAILABLE_VOICES,
)
from app.core.logging import logger

router = APIRouter(prefix="/tts", tags=["tts"])


@router.get("/voices", response_model=TTSVoicesResponse)
async def list_voices():
    return TTSVoicesResponse(
        voices=[TTSVoiceInfo(name=v["name"], language=v["language"]) for v in AVAILABLE_VOICES],
        sample_rate=get_sample_rate(),
    )


@router.post("/synthesize")
async def synthesize(req: TTSRequest):
    if not is_tts_loaded():
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    try:
        model = get_tts_model()
        voice_state = get_voice_state(req.voice)

        if voice_state is None:
            voice_state = model.get_state_for_audio_prompt(req.voice)

        audio = model.generate_audio(voice_state, req.text)

        audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio

        buffer = io.BytesIO()
        sample_rate = get_sample_rate()
        scipy.io.wavfile.write(buffer, sample_rate, audio_np)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_{req.voice}.wav"
            }
        )
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))