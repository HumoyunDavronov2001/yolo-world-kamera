"""Noutbuk kamerasi + YOLO-World: 1000 ta sinfni real vaqtda aniqlash.

Tugmalar:  q / Esc - chiqish   + / - - ishonch chegarasi   s - kadrni saqlash
Sinov:     python kamera.py --sinov 30   (oynasiz 30 kadr, sinov_natija.jpg ga yozadi)
"""

import argparse
import os
import threading
import time
import zlib
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLOWorld

JOY = Path(__file__).parent
torch.set_num_threads(os.cpu_count())


class Kamera:
    """Kadrlarni alohida oqimda o'qiydi: model sekin bo'lsa ham doim eng yangi kadr olinadi (kechikish to'planmaydi)."""

    def __init__(self, indeks, en, boy):
        self.cap = cv2.VideoCapture(indeks, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, en)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, boy)
        if not self.cap.isOpened():
            raise SystemExit(f"Kamera ochilmadi: /dev/video{indeks}")
        self.kadr, self.ishla = None, True
        self.qulf = threading.Lock()
        threading.Thread(target=self._oqi, daemon=True).start()

    def _oqi(self):
        while self.ishla:
            ok, f = self.cap.read()
            if ok:
                with self.qulf:
                    self.kadr = f

    def ol(self):
        while True:
            with self.qulf:
                if self.kadr is not None:
                    return self.kadr.copy()
            time.sleep(0.01)

    def yop(self):
        self.ishla = False
        self.cap.release()


def rang(i):
    h = zlib.crc32(str(i).encode())
    return (h & 255, (h >> 8) & 255, 80 + ((h >> 16) & 175))


def chiz(f, natija, nomlar, fps, conf):
    sanoq = Counter()
    for (x1, y1, x2, y2), c, p in zip(natija.boxes.xyxy.int().tolist(), natija.boxes.cls.int().tolist(), natija.boxes.conf.tolist()):
        r = rang(c)
        sanoq[nomlar[c]] += 1
        cv2.rectangle(f, (x1, y1), (x2, y2), r, 2)
        yozuv = f"{nomlar[c]} {p:.2f}"
        (w, h), _ = cv2.getTextSize(yozuv, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        yt = max(y1, h + 6)
        cv2.rectangle(f, (x1, yt - h - 6), (x1 + w + 6, yt), r, -1)
        cv2.putText(f, yozuv, (x1 + 3, yt - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Chap yuqori: holat paneli va eng ko'p uchragan sinflar
    qatorlar = [f"FPS {fps:4.1f} | 1000 sinf | obyekt: {sum(sanoq.values())} | ishonch >= {conf:.2f}"]
    qatorlar += [f"{n}: {k}" for n, k in sanoq.most_common(12)]
    panel = f.copy()
    cv2.rectangle(panel, (0, 0), (430, 14 + 22 * len(qatorlar)), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, f, 0.45, 0, f)
    for i, q in enumerate(qatorlar):
        cv2.putText(f, q, (10, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return sanoq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kamera", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=480, help="kattaroq = aniqroq, lekin sekinroq (640)")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--sinov", type=int, default=0, help="oynasiz N kadr ishlab, natijani faylga yozadi")
    a = ap.parse_args()

    model = YOLOWorld(str(JOY / "world_1000.pt"))
    nomlar = model.names
    print(f"Model: {len(nomlar)} ta sinf, imgsz={a.imgsz}")
    kam = Kamera(a.kamera, 1280, 720)
    oyna = "YOLO-World | 1000 sinf"
    if not a.sinov:
        cv2.namedWindow(oyna, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(oyna, 1280, 720)

    conf, fps, n, vaqtlar = a.conf, 0.0, 0, []
    try:
        while True:
            t = time.time()
            f = kam.ol()
            natija = model.predict(f, imgsz=a.imgsz, conf=conf, agnostic_nms=True, max_det=100, verbose=False)[0]
            vaqtlar.append(time.time() - t)
            fps = 0.8 * fps + 0.2 / vaqtlar[-1] if fps else 1 / vaqtlar[-1]
            sanoq = chiz(f, natija, nomlar, fps, conf)
            n += 1

            if a.sinov:
                if n >= a.sinov:
                    cv2.imwrite(str(JOY / "sinov_natija.jpg"), f)
                    o = sorted(vaqtlar[3:])
                    print(f"{n} kadr, o'rtacha {sum(o) / len(o) * 1000:.0f} ms, median {o[len(o) // 2] * 1000:.0f} ms, FPS ~{len(o) / sum(o):.1f}")
                    print("oxirgi kadr:", dict(sanoq))
                    break
                continue

            cv2.imshow(oyna, f)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27) or cv2.getWindowProperty(oyna, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k in (ord("+"), ord("=")):
                conf = min(0.9, conf + 0.05)
            elif k in (ord("-"), ord("_")):
                conf = max(0.05, conf - 0.05)
            elif k == ord("s"):
                p = JOY / f"kadr_{time.strftime('%H%M%S')}.jpg"
                cv2.imwrite(str(p), f)
                print("saqlandi:", p)
    finally:
        kam.yop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
