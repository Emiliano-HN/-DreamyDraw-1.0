#🎨 Bienvenido a DreamyDraw 🐣

# DreamyDraw 🐣 es una inteligencia creativa diseñada para transformar tus ideas en imágenes únicas.
# - ✨ Solo escribe lo que quieras imaginar, y DreamyDraw dibujará tu sueño.
# - 🚀 Si tu computadora tiene GPU, las imágenes se generarán más rápido.
# - 🐢 Si usa solo CPU, tardará un poco más, pero no te preocupes: ¡la espera vale la pena!

#🎯 Recuerda:
# - Cada imagen es única y creada con dedicación.
# - DreamyDraw 🐣 es experimental y puede requerir unos momentos para plasmar tu visión.

# Creado con pasión por Emiliano Hernández Navarrete.

!pip install diffusers transformers accelerate scipy safetensors --quiet

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from google.colab import files

modelo = "stabilityai/stable-diffusion-2-1"

print(" Cargando DreamyDraw 🐣, paciencia... 🚀")

if torch.cuda.is_available():
    pipe = StableDiffusionPipeline.from_pretrained(
        modelo,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        modelo,
        torch_dtype=torch.float32,
        use_safetensors=True
    ).to("cpu")

print(f" DreamyDraw 🐣 cargado en {pipe.device} correctamente.")

def dreamy_draw(prompt, nombre_archivo="imagen_dreamy.png"):
    print(f" DreamyDraw 🐣 Generando tu sueño: {prompt}")

    image = pipe(prompt).images[0]

    image.save(nombre_archivo)
    print(f"Imagen lista en DreamyDraw 🐣: '{nombre_archivo}'")

    files.download(nombre_archivo)

prompt = input("🎨 ¿Qué quieres soñar hoy con DreamyDraw 🐣?: ")
dreamy_draw(prompt)
