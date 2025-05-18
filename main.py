
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OCR_API_KEY = os.getenv("OCR_API_KEY")

HUGGINGFACE_API_KEY = "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"
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"

def analizar_con_gptneo(texto):
    prompt = f"Analiza esta respuesta de estudiante: '{texto}'. Evalúa si es correcta, qué habilidades demuestra, errores y sugiere retroalimentación pedagógica. Devuelve el análisis completo en español."

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 300
        }
    }

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            return f"Error en HuggingFace API (status {response.status_code}): {response.text}"
        resultado = response.json()
        if isinstance(resultado, list) and "generated_text" in resultado[0]:
            return resultado[0]["generated_text"]
        elif "generated_text" in resultado:
            return resultado["generated_text"]
        else:
            return str(resultado)
    except Exception as e:
        return f"Error en llamada a HuggingFace: {str(e)}"

@app.post("/evaluar")
async def evaluar(file: UploadFile = File(...)):
    try:
        ocr_response = requests.post(
            "https://api.ocr.space/parse/image",
            data={"apikey": OCR_API_KEY, "language": "spa"},
            files={"file": (file.filename, await file.read(), file.content_type)},
        )
        resultado_ocr = ocr_response.json()
        texto_extraido = resultado_ocr["ParsedResults"][0]["ParsedText"]
    except Exception:
        return {"error": "❌ No se pudo leer texto desde la imagen."}

    analisis = analizar_con_gptneo(texto_extraido)

    return {
        "texto_extraido": texto_extraido,
        "analisis_gpt": analisis
    }
