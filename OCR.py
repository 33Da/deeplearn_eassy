
import os
import pytesseract

from PIL import Image

# imgs_save_path = "D:\\PythonProjects\\test_conda\\检测\\download\\"
# txts_save_path = "D:\\PythonProjects\\test_conda\\检测\\download\\"

# im_path = r'D:\\PythonProjects\\test_conda\\检测\\download\\img_10112.png'
# 识别单张:
def img_to_txt(imgs_path, save_path):
    # for files, _, file_names in os.walk(imgs_path):
    #     print("files:",files)
    #     print("file_names:",file_names)
    #     for file_name in file_names:
    image = Image.open(imgs_path)
    # chi_sim 是中文识别包，equ 是数学公式包，eng 是英文包
    content = pytesseract.image_to_string(image, lang="chi_sim")
    # print(content)
    txt_name = imgs_path.split('\\')[-1].split('/')[-1][:-4] + ".txt"
    with open(save_path +'\\'+txt_name, "a+",encoding='utf-8') as f:
        f.write(content)
    return     content
# 识别多张
def img_to_txts(imgs_path, save_path):
    # 将文件夹下的所有图片进行OCR识别
            image = Image.open(imgs_path)
            # chi_sim 是中文识别包，equ 是数学公式包，eng 是英文包
            content = pytesseract.image_to_string(image, lang="chi_sim")
            print(content)
            return    content

# img_to_txt(imgs_save_path, txts_save_path)
