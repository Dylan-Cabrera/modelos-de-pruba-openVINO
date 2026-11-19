"""Exporta YOLOE-26 a OpenVINO con las clases del laboratorio ya fijadas.

YOLOE es de vocabulario abierto: las clases se le dan en texto y un codificador de texto
(MobileCLIP2, ~254 MB) las convierte en embeddings. Fijarlas antes de exportar deja esos
embeddings dentro del modelo, así vortex_stream.py no carga el codificador en cada
arranque y corre en OpenVINO como cualquier YOLO.

Uso:
    python export_yoloe.py                       # s y m, a 640
    python export_yoloe.py --sizes m --imgsz 640,960
    python export_yoloe.py --classes "person,face mask,lab coat"

Salida, por tamaño de modelo y de entrada:
    yoloe-26s-seg-lab_openvino_model/        (imgsz 640)
    yoloe-26s-seg-lab-960_openvino_model/    (otro imgsz lleva el número en el nombre)

Los pesos (.pt, .ts) y las carpetas exportadas no se versionan (.gitignore).
"""

import argparse
import os
import shutil
import sys

from ultralytics import YOLOE

DEFAULT_CLASSES = "person,face mask,white lab coat"


def parse_args():
    parser = argparse.ArgumentParser(description="Exporta YOLOE-26 a OpenVINO con clases fijas")
    parser.add_argument(
        "--sizes",
        type=str,
        default="s,m",
        help="Tamaños de YOLOE-26 a exportar, separados por coma (default: 's,m')",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=DEFAULT_CLASSES,
        help=f"Clases en texto, separadas por coma (default: '{DEFAULT_CLASSES}')",
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default="640",
        help="Tamaños de entrada a exportar, separados por coma (default: '640')",
    )
    return parser.parse_args()


def output_dir(size: str, imgsz: int) -> str:
    # 640 es el tamaño por defecto de vortex_stream.py: su carpeta va sin número para
    # que el nombre corto sea el que se usa siempre.
    suffix = "" if imgsz == 640 else f"-{imgsz}"
    return f"yoloe-26{size}-seg-lab{suffix}_openvino_model"


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    imgszs = [int(s) for s in args.imgsz.split(",") if s.strip()]
    if not classes or not sizes or not imgszs:
        sys.exit("Error: --classes, --sizes e --imgsz no pueden quedar vacíos.")

    for size in sizes:
        weights = f"yoloe-26{size}-seg.pt"
        for imgsz in imgszs:
            target = output_dir(size, imgsz)
            print(f"==> {weights} @ {imgsz} con clases {classes} -> {target}")
            # Se recarga en cada exportación para que cada carpeta salga de los pesos
            # originales y no del modelo ya fusionado por la exportación anterior.
            model = YOLOE(weights)
            model.set_classes(classes)
            exported = model.export(format="openvino", imgsz=imgsz, half=True)
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.move(exported, target)
            print(f"    listo: {target}")


if __name__ == "__main__":
    main()
