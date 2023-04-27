from PIL import Image
import tabula
import pandas as pd
import fitz
import PIL.Image
import io

def all_images_from_pdf (file):
    pdf = fitz.open(file)
    counter = 0
    for i in range(len(pdf)):
        page = pdf[i]
        images = page.get_images()
        for image in images:
            base_img = pdf.extract_image(image[0])
            image_data = base_img["image"]
            img = PIL.Image.open(io.BytesIO(image_data))
            extension = base_img["ext"]
            img.save(open(f"image{counter}.png", "wb"))
            counter += 1

if __name__ == '__main__':

    tables = tabula.read_pdf("file.pdf", pages="all")
    if tables:
        for i in range (len(tables)):
            tables[i].to_csv('table'+str(i)+'clear_table_in_pdf'+'.csv', index=False)

    all_images_from_pdf("file.pdf")