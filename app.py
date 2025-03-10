import streamlit as st
from doctr.io import DocumentFile
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
import pytesseract

import utils

# Register a Unicode-compatible font
fontname = "Ubuntu"
fontpath = "./Ubuntu-Regular.ttf"
reco_arch = "kz_latest.pt"
pdfmetrics.registerFont(TTFont(fontname, fontpath))


def main():
    """Построение интерфейса Streamlit"""
    # Широкий режим
    st.set_page_config(layout="wide")

    # Дизайн интерфейса
    st.title("Құжаттағы мәтінді тану")
    # Новая строка
    st.write("\n")
    # Инструкции
    st.markdown(
        "*Кеңес: суреттің оң жақ жоғарғы бұрышын бассаңыз, оны үлкейте аласыз!*"
    )
    # Установка колонок
    cols = st.columns((1, 1, 1))
    cols[0].subheader("Бастапқы бет")
    cols[1].subheader("OCR нәтижесі")
    cols[2].subheader("Мәтіннің біріктірілген нұсқасы")

    # Боковая панель
    # Выбор файла
    st.sidebar.title("Құжатты таңдау")
    # Загрузка собственного изображения
    uploaded_file = st.sidebar.file_uploader(
        "Файлдарды жүктеңіз", type=["pdf", "png", "jpeg", "jpg"]
    )
    use_pytesseract = st.sidebar.checkbox("Пайдалану pytesseract арқылы OCR", False)
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            doc = DocumentFile.from_pdf(uploaded_file.read())
        else:
            doc = DocumentFile.from_images(uploaded_file.read())
        page_idx = (
            st.sidebar.selectbox("Бетті таңдау", [idx + 1 for idx in range(len(doc))])
            - 1
        )
        page = doc[page_idx]
        cols[0].image(page)

        with st.spinner("Модельді жүктеу..."):
            predictor = utils.get_ocr_predictor(
                reco_arch=reco_arch,
            )

        with st.spinner("Талдау..."):
            # Пропуск изображения через модель
            out = predictor([page])
            page_export = out.pages[0].export()

            (coordinates, _, _) = utils.page_to_coordinates(page_export)

            boxes_with_labels = utils.draw_boxes_with_labels(
                page, coordinates, font_path="./Ubuntu-Regular.ttf"
            )
            cols[1].image(boxes_with_labels)

            # Отображение объединенного текста
            final_text = utils.ocr_to_txt(coordinates)
            cols[2].text_area("Мәтіннің біріктірілген нұсқасы:", final_text, height=500)

            # Use pytesseract if checkbox is selected
            if use_pytesseract:
                if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(uploaded_file)
                    ocr_text = pytesseract.image_to_string(image, lang="kaz+eng+rus")

                    # Create a collapsible block for OCR results
                    with st.expander("OCR нәтижесі (pytesseract)"):
                        st.text_area("Тексеру нәтижесі:", ocr_text, height=300)
                else:
                    st.warning("OCR тек суреттер үшін қол жетімді.")

            
            st.markdown("\nТалдау нәтижелері JSON форматында берілген:")
            st.json(page_export, expanded=False)


if __name__ == "__main__":
    main()
