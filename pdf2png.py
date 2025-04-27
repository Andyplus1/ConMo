# pdf_to_png.py

from pdf2image import convert_from_path
import os

# 设置输入和输出
pdf_path = "pic_final_2.pdf"       # 你的 PDF 文件名
output_folder = "./"  # 输出文件夹名
dpi = 300                        # 图片清晰度，通常 300 比较高清

def pdf_to_png(pdf_path, output_folder, dpi=300):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 将 PDF 转成图片
    images = convert_from_path(pdf_path, dpi=dpi)

    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"page_{i+1}.png")
        image.save(output_path, 'PNG')
        print(f"Saved {output_path}")

if __name__ == "__main__":
    pdf_to_png(pdf_path, output_folder)
