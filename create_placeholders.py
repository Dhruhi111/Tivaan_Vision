# create_placeholders.py
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
STATIC_RESULTS = Path("static") / "results"
STATIC_RESULTS.mkdir(parents=True, exist_ok=True)

def make_text_image(path, txt, size=(800,400), bgcolor=(30,30,30), fg=(220,220,220)):
    img = Image.new("RGB", size, bgcolor)
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("arial.ttf", 32)
    except:
        f = ImageFont.load_default()
    w,h = d.textsize(txt, font=f)
    d.text(((size[0]-w)/2,(size[1]-h)/2), txt, font=f, fill=fg)
    img.save(path)

make_text_image(STATIC_RESULTS / "labels.jpg", "labels.jpg (placeholder)")
make_text_image(STATIC_RESULTS / "confusion_matrix.png", "confusion_matrix.png (placeholder)")
make_text_image(STATIC_RESULTS / "results.png", "results.png (placeholder)")
print("placeholders created at", STATIC_RESULTS)
