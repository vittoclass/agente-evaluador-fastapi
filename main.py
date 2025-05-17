
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import openai
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OCR_API_KEY = os.getenv("OCR_API_KEY")
GPT_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = GPT_API_KEY

@app.post("/evaluar")
async def evaluar(file: UploadFile = File(...)):
    ocr_response = requests.post(
        "https://api.ocr.space/parse/image",
        data={"apikey": OCR_API_KEY, "language": "spa"},
        files={"file": (file.filename, await file.read(), file.content_type)},
    )
    resultado_ocr = ocr_response.json()
    try:
        texto_extraido = resultado_ocr["ParsedResults"][0]["ParsedText"]
    except Exception:
        return {"error": "No se pudo leer la imagen correctamente"}

    prompt = f"""
    Analiza esta respuesta escrita por un estudiante. Indica si es correcta, qué habilidades demuestra, qué errores tiene (si hay), y sugiere una retroalimentación pedagógica. Respuesta del estudiante:
    "{texto_extraido}"
    Devuelve el resultado en JSON con los siguientes campos: correcto (sí/no), habilidades, errores, retroalimentación y puntaje (de 1 a 7).
    """

    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return {
        "texto_extraido": texto_extraido,
        "analisis_gpt": gpt_response["choices"][0]["message"]["content"]
    }
