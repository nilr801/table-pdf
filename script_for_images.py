import io
import os

import PIL.Image
import cv2
import fitz
import layoutparser as lp
import pandas as pd
from PIL import Image
import pytesseract

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def cropp_image(img, annotation):
    x_min, y_min = int(annotation[0]), int(annotation[1])
    x_max, y_max = int(annotation[2]), int(annotation[3])
    return img.crop((x_min, y_min, x_max, y_max))



def draw_bounding_box(img, annotation):
    x_min, y_min = int(annotation[0]), int(annotation[1])
    x_max, y_max = int(annotation[2]), int(annotation[3])

    color = (234, 150, 105)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

def table_from_img(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    table_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            table_contour = contour

    x,y,w,h = cv2.boundingRect(table_contour)
    table_img = img[y:y+h,x:x+w]


    gray_table = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    thresh_table = cv2.threshold(gray_table, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(thresh_table, lang='eng',config='--psm 6')

    rows = text.split('\n')
    table_data = []
    for row in rows:
        table_data.append(row.split())
    df = pd.DataFrame(table_data)
    return df



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

    image_path = 'input/image1.png'
    image = cv2.imread(image_path)
    table_model = lp.Detectron2LayoutModel('lp://TableBank/faster_rcnn_R_101_FPN_3x/config',
                                           extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6],
                                           label_map={0: "Table"})
    images_tables = {}
    layout = table_model.detect(image)
    height, weight = image.shape[:2]
    tables = []
    for table_index in range(len(layout._blocks)):
        x_1, y_1, x_2, y_2 = layout._blocks[table_index].coordinates
        tables.append((x_1, y_1, x_2, y_2))

    image = Image.open(image_path)
    for i in range(len(tables)):
        cropp_image(image, tables[i]).save("cropped_example" + str(i) + ".png")
    for i in range(len(tables)):
        table_from_img("cropped_example" + str(i) + ".png").to_csv('table'+str(i)+'.csv', index=False)
