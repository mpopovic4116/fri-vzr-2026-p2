from argparse import ArgumentParser

from PIL import Image, ImageChops


def cmp(img1, img2):
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    frame = 0
    try:
        while True:
            diff = ImageChops.difference(img1.convert("RGB"), img2.convert("RGB"))
            if diff.getbbox():
                print(f"MISSMATCH AT FRAME: {frame}")
                return False

            frame += 1
            img1.seek(img1.tell() + 1)
            img2.seek(img2.tell() + 1)
    except Exception:  # noqa: BLE001
        print("Gifs are identical.")
        return True


def main():
    parser = ArgumentParser()
    parser.add_argument("img1")
    parser.add_argument("img2")
    parsed = parser.parse_args()

    cmp(parsed.img1, parsed.img2)
