from fastapi import FastAPI, Request, UploadFile, File, Form
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from safetensors.torch import load_file

import pytesseract
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from typing import List, Optional
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()
# ====================== CLASE ======================
class RobertaMultitaskClassifier(nn.Module):
    def __init__(self, model_name, num_labels_age=3, num_labels_gender=2, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True
        )
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.encoder.config.hidden_size
        self.classifier_age = nn.Linear(hidden_size, num_labels_age)
        self.classifier_gender = nn.Linear(hidden_size, num_labels_gender)

    def forward(self, input_ids, attention_mask, labels_age=None, labels_gender=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits_age = self.classifier_age(pooled)
        logits_gender = self.classifier_gender(pooled)

        loss = None
        if labels_age is not None and labels_gender is not None:
            loss_age = nn.CrossEntropyLoss()(logits_age, labels_age)
            loss_gender = nn.CrossEntropyLoss()(logits_gender, labels_gender)
            loss = loss_age + loss_gender

        return {"loss": loss, "logits_age": logits_age, "logits_gender": logits_gender}


# ====================== MODELO 1: Formalidad ======================
modelo_formal = AutoModelForSequenceClassification.from_pretrained("modelo_formalidad").to("cuda")
tokenizer_formal = AutoTokenizer.from_pretrained("modelo_formalidad")

# ====================== MODELO 2: Agresividad ======================
modelo_agresivo = AutoModelForSequenceClassification.from_pretrained("modelo_agresividad").to("cuda")
tokenizer_agresivo = AutoTokenizer.from_pretrained("modelo_agresividad")

# ====================== MODELO 3: Público objetivo ======================
tokenizer_publico = AutoTokenizer.from_pretrained("modelo_publico")
modelo_publico = RobertaMultitaskClassifier(
    model_name="PlanTL-GOB-ES/roberta-base-bne",
    num_labels_age=3,
    num_labels_gender=2
)
state_dict = load_file("modelo_publico/model.safetensors", device="cuda")
modelo_publico.load_state_dict(state_dict)
modelo_publico.to("cuda").eval()

# ====================== FUNCIONES DE ANÁLISIS ======================
def analizar_texto(texto_total):
    inputs_f = tokenizer_formal(texto_total, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")
    with torch.no_grad():
        out_f = modelo_formal(**inputs_f)
        probs_f = F.softmax(out_f.logits, dim=1)[0].tolist()
        pred_f = torch.argmax(out_f.logits, dim=1).item()

    inputs_a = tokenizer_agresivo(texto_total, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")
    with torch.no_grad():
        out_a = modelo_agresivo(**inputs_a)
        probs_a = F.softmax(out_a.logits, dim=1)[0].tolist()
        pred_a = torch.argmax(out_a.logits, dim=1).item()

    inputs_p = tokenizer_publico(texto_total, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs_p = {k: v.to("cuda") for k, v in inputs_p.items()}
    with torch.no_grad():
        out_p = modelo_publico(**inputs_p)
        probs_age = F.softmax(out_p["logits_age"], dim=1)[0].tolist()
        probs_gender = F.softmax(out_p["logits_gender"], dim=1)[0].tolist()
        pred_age = torch.argmax(out_p["logits_age"], dim=1).item()
        pred_gender = torch.argmax(out_p["logits_gender"], dim=1).item()

    return {
        "formalidad": {"label": pred_f, "probs": probs_f},
        "agresividad": {"label": pred_a, "probs": probs_a},
        "publico_objetivo": {
            "edad": ["18-29", "30-39", "40-49"][pred_age],
            "genero": ["male", "female"][pred_gender],
            "probs_edad": probs_age,
            "probs_genero": probs_gender
        }
    }


def preprocesar_imagen(imagen_pil: Image.Image) -> Image.Image:
    # Convertir PIL a OpenCV (BGR)
    imagen_np = np.array(imagen_pil)
    if imagen_np.ndim == 3 and imagen_np.shape[2] == 4:  # RGBA
        imagen_np = cv2.cvtColor(imagen_np, cv2.COLOR_RGBA2BGR)
    else:
        imagen_np = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)

    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_BGR2GRAY)

    # Aplicar suavizado (opcional, ayuda con el ruido)
    suavizado = cv2.GaussianBlur(gris, (3, 3), 0)

    # Binarización adaptativa (útil para condiciones de luz variables)
    binarizada = cv2.adaptiveThreshold(
        suavizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Otras opciones posibles:
    # _, binarizada = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Opcional: invertir colores si el texto es blanco sobre negro
    # binarizada = cv2.bitwise_not(binarizada)

    # Convertir de nuevo a PIL para pytesseract
    imagen_procesada = Image.fromarray(binarizada)
    return imagen_procesada

# ====================== SCHEMAS ======================
class TextInput(BaseModel):
    texto: str

class TextListInput(BaseModel):
    textos: list[str]

# ====================== RUTA INDIVIDUAL ======================
@app.post("/analizar")
def analizar_texto(entrada: TextInput):
    return analizar_batch(TextListInput(textos=[entrada.texto]))[0]

# ====================== RUTA BATCH ======================
@app.post("/analizar_batch")
def analizar_batch(entrada: TextListInput):
    resultados = []

    # ------ Formalidad ------
    inputs_f = tokenizer_formal(entrada.textos, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")
    with torch.no_grad():
        out_f = modelo_formal(**inputs_f)
        probs_f = F.softmax(out_f.logits, dim=1)
        preds_f = torch.argmax(probs_f, dim=1)

    # ------ Agresividad ------
    inputs_a = tokenizer_agresivo(entrada.textos, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")
    with torch.no_grad():
        out_a = modelo_agresivo(**inputs_a)
        probs_a = F.softmax(out_a.logits, dim=1)
        preds_a = torch.argmax(probs_a, dim=1)

    # ------ Público objetivo ------
    inputs_p = tokenizer_publico(entrada.textos, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs_p = {k: v.to("cuda") for k, v in inputs_p.items()}
    with torch.no_grad():
        out_p = modelo_publico(**inputs_p)
        probs_age = F.softmax(out_p["logits_age"], dim=1)
        probs_gender = F.softmax(out_p["logits_gender"], dim=1)
        preds_age = torch.argmax(probs_age, dim=1)
        preds_gender = torch.argmax(probs_gender, dim=1)

    for i, texto in enumerate(entrada.textos):
        resultados.append({
            "texto": texto,
            "formalidad": {
                "label": preds_f[i].item(),
                "probs": probs_f[i].tolist()
            },
            "agresividad": {
                "label": preds_a[i].item(),
                "probs": probs_a[i].tolist()
            },
            "publico_objetivo": {
                "edad": ["18-29", "30-39", "40-49"][preds_age[i].item()],
                "genero": ["male", "female"][preds_gender[i].item()],
                "probs_edad": probs_age[i].tolist(),
                "probs_genero": probs_gender[i].tolist()
            }
        })

    return resultados

# ====================== RUTA INDIVIDUAL ======================
@app.post("/analizar_full")
async def analizar_full(
    texto: Optional[str] = Form(None),
    imagen: Optional[UploadFile] = File(None)
):
    resultado = {}

    if texto and texto.strip():
        resultado["texto_analizado"] = {
            "descripcion": texto.strip(),
            "resultados": analizar_texto(TextInput(texto=texto.strip()))
        }

    
    if imagen:
        try:
            contenido = await imagen.read()
            # imagen_pil = Image.open(BytesIO(contenido)).convert("RGB")
            imagen_pil = Image.open(BytesIO(contenido))
            imagen_preprocesada = preprocesar_imagen(imagen_pil)

            extraido = pytesseract.image_to_string(imagen_preprocesada, lang="spa").strip()

            print(f"[OCR] Imagen procesada - Texto extraído:\n{extraido}\n")
            # print(f"[OCR] Imagen procesada - Texto extraído")

            if extraido:
                resultado["texto_imagen"] = {
                    "descripcion": extraido,
                    "resultados": analizar_texto(TextInput(texto=extraido))
                }
            else:
                resultado["texto_imagen"] = {
                    "error": "OCR no extrajo texto"
                }
        except UnidentifiedImageError as e:
            # print(f"[ERROR OCR] No se pudo abrir la imagen: {e}")
            print(f"[ERROR OCR] No se pudo abrir la imagen")
            resultado["texto_imagen"] = {
                "error": "Imagen no válida o corrupta"
            }
        except Exception as e:
            # print(f"[ERROR OCR] Excepción inesperada: {e}")
            print(f"[ERROR OCR] Excepción inesperada")
            resultado["texto_imagen"] = {
                "error": "Fallo inesperado al procesar la imagen",
                "detalle": str(e)
            }

    if not resultado:
        resultado["sin_contenido"] = True

    return resultado
