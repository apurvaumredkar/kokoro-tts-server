import io
import re

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import onnxruntime as ort
from kokoro_onnx import Kokoro
import soundfile as sf

app = FastAPI()

# Emoji, pictographs, dingbats, symbols, flags, variation selectors and joiners.
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # pictographs, emoticons, transport, symbols, flags
    "\U00002190-\U000021FF"  # arrows
    "\U00002300-\U000023FF"  # misc technical
    "\U000025A0-\U000027BF"  # shapes, misc symbols, dingbats
    "\U00002900-\U0000297F"  # supplemental arrows
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "\U0000FE0E\U0000FE0F\U0000200D\U000020E3"  # variation selectors, joiner, keycap
    "]+"
)
_FENCE = re.compile(r"```.*?```", re.DOTALL)  # code is not read aloud
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_QUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_RULE = re.compile(r"^\s*([-*_]\s*){3,}$", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_BOLD_ITALIC = re.compile(r"(\*{1,3})(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_UNDERSCORE = re.compile(r"(?<!\w)(_{1,3})(?=\S)(.+?)(?<=\S)\1(?!\w)", re.DOTALL)
_STRIKE = re.compile(r"~~(.+?)~~", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_TABLE_SEP = re.compile(r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$", re.MULTILINE)
_TABLE_BAR = re.compile(r"\|")
_SPACES = re.compile(r"[ \t]+")


def speakable(text: str) -> str:
    """Strip markdown and emoji so the TTS only reads words."""
    text = _FENCE.sub(" ", text)
    text = _IMAGE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _HEADING.sub("", text)
    text = _QUOTE.sub("", text)
    text = _RULE.sub("", text)
    text = _BULLET.sub("", text)
    text = _BOLD_ITALIC.sub(r"\2", text)
    text = _UNDERSCORE.sub(r"\2", text)
    text = _STRIKE.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _TABLE_SEP.sub("", text)
    text = _TABLE_BAR.sub(" ", text)
    text = text.replace("*", "").replace("`", "")  # leftovers from unbalanced marks
    text = _EMOJI.sub("", text)
    text = _SPACES.sub(" ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()

MODEL = "models/kokoro-v1.0.onnx"
VOICES = "models/voices-v1.0.bin"

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

providers = [
    (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            "cudnn_conv_algo_search": "DEFAULT",
        },
    ),
    "CPUExecutionProvider",
]

session = ort.InferenceSession(MODEL, sess_options=so, providers=providers)
print("Active providers:", session.get_providers())


kokoro = Kokoro.from_session(session, VOICES)


@app.get("/health")
async def health():
    return {"status": "ok"}


class GenerateRequest(BaseModel):
    text: str
    voice: str = "bm_daniel"
    speed: float = 1.0
    lang: str = "en-us"


@app.post("/generate")
def generate_voice(req: GenerateRequest):
    text = speakable(req.text)
    if not text:
        raise HTTPException(400, "nothing to say after removing markdown and emoji")
    samples, sr = kokoro.create(text, voice=req.voice, speed=req.speed, lang=req.lang)
    buffer = io.BytesIO()
    sf.write(buffer, samples, sr, format="WAV")
    return Response(content=buffer.getvalue(), media_type="audio/wav")
