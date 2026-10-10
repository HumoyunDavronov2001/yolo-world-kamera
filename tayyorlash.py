"""1000 ta sinf ro'yxatini yasaydi va YOLO-World modeliga bir marta "o'rnatib" saqlaydi.

Tartib: COCO (80) -> Objects365 (365) -> LVIS (1203), takrorlar olib tashlanadi, birinchi 1000 tasi olinadi.
Shu tartib kundalik buyumlarni birinchi o'ringa qo'yadi. CLIP faqat shu yerda ishlaydi;
kamera.py tayyor world_1000.pt ni yuklaydi va CLIP'ga muhtoj emas (xotira tejaladi).
"""

import re
from pathlib import Path

import yaml
import ultralytics
from ultralytics import YOLOWorld

SON = 1000
DS = Path(ultralytics.__file__).parent / "cfg" / "datasets"
JOY = Path(__file__).parent


def nomlar(fayl):
    return list(yaml.safe_load((DS / fayl).read_text())["names"].values())


def tozala(nom):
    nom = nom.split("/")[0]  # LVIS: "airplane/aeroplane" -> "airplane"
    nom = re.sub(r"^other\s+", "", nom, flags=re.I)  # Objects365: "Other Shoes" -> "Shoes"
    nom = nom.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", nom)


def kalit(nom):
    # "shoes" va "shoe" bir xil sinf hisoblanadi
    k = re.sub(r"[^a-z0-9 ]", "", nom)
    return k[:-1] if k.endswith("s") and not k.endswith("ss") else k


sinflar, korilgan = [], set()
for fayl in ("coco.yaml", "Objects365.yaml", "lvis.yaml"):
    for n in nomlar(fayl):
        n = tozala(n)
        if n and kalit(n) not in korilgan:
            korilgan.add(kalit(n))
            sinflar.append(n)

sinflar = sinflar[:SON]
assert len(sinflar) == SON, len(sinflar)
(JOY / "sinflar.txt").write_text("\n".join(sinflar) + "\n")
print(f"{len(sinflar)} ta sinf -> sinflar.txt")

model = YOLOWorld(str(JOY / "yolov8s-worldv2.pt"))
model.set_classes(sinflar)
model.model.clip_model = None  # CLIP saqlanmasin: fayl kichik, kamera.py da CLIP yuklanmaydi
model.save(str(JOY / "world_1000.pt"))
print("saqlandi: world_1000.pt")
