import os
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

def add_numbering_to_image(image_path, number):
    img = Image.open(image_path).convert("RGB")
    new_size = (int(img.width * 1.5), int(img.height * 1.5))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = str(number)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x, y = 10, 10
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return img

def create_space_image(prev_img_size):
    width, height = prev_img_size
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = 'space'
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, font=font, fill=(255, 0, 0))
    return img

def generate_gif_from_sentence(sentence):
    images = []
    prev_img_size = (150, 150)

    for index, char in enumerate(sentence):
        char_upper = char.upper()

        if char == ' ':
            space_image = create_space_image(prev_img_size)
            images.append(space_image)
        else:
            char_folder = os.path.join("dataset", char_upper)
            if os.path.isdir(char_folder):
                image_files = sorted([
                    f for f in os.listdir(char_folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                if image_files:
                    first_image_path = os.path.join(char_folder, image_files[0])
                    numbered_image = add_numbering_to_image(first_image_path, index + 1)
                    prev_img_size = numbered_image.size
                    images.append(numbered_image)

    if images:
        gif_path = "output.gif"
        images[0].save(gif_path, save_all=True, append_images=images[1:], loop=0, duration=800)
        return gif_path
    return None

st.title("Sentence to Sign Language GIF")
sentence = st.text_input("Enter a sentence:")

if sentence:
    gif_path = generate_gif_from_sentence(sentence)
    if gif_path:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(gif_path, caption="Generated GIF", use_column_width=False)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("No valid images found for the entered sentence.")





