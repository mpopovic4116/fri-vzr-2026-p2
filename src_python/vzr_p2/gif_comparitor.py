from PIL import Image, ImageChops


def cmp(img1, img2):
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    frame = 0
    try:
        while True:
            diff = ImageChops.difference(
                img1.convert("RGB"), img2.convert("RGB"))
            if diff.getbbox():
                print(f"MISSMATCH AT FRAME: {frame}")
                return False

            frame += 1
            img1.seek(img1.tell() + 1)
            img2.seek(img2.tell() + 1)
    except Exception:
        print(f"Gifs are identical.")
        return True


cmp("lenia_cpu.gif", "lenia.gif")
