# from huggingface_hub import InferenceClient

# # Correction model client
# correction_client = InferenceClient(
#     provider="sambanova",
#     api_key=""
# )

# # Translation model client
# translation_client = InferenceClient(
#     provider="hf-inference",
#     api_key=""
# )

# def correct_sentence(s: str) -> str:
#     prompt = f'correct the following sentence with correct words and spacings, only give corrected sentence in "" and nothing else: {s}'
#     completion = correction_client.chat.completions.create(
#         model="deepseek-ai/DeepSeek-V3-0324",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return completion.choices[0].message.content.strip().replace('"', '')

# def translate_to_hindi(text: str) -> str:
#     result = translation_client.translation(
#         text,
#         model="Helsinki-NLP/opus-mt-en-hi"
#     )
#     # Strip known prefix if it exists
#     return result.translation_text.replace("यहाँ सही वाक्य है: ", "").strip()

# # Example usage
# input_text = "HBLOH0RAreY00"
# corrected = correct_sentence(input_text)
# print(corrected)

# translated = translate_to_hindi(corrected)
# print(translated)









import google.generativeai as genai

# Correct way to configure your API key
genai.configure(api_key="")

# Choose your model (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to correct sentence
def correct_sentence(s: str) -> str:
    prompt = f"Use your knowledge and predict/Correct the following sentence with correct words and spacings. Only give corrected sentence:\n{s}"
    response = model.generate_content(prompt)
    return response.text.strip()

# Function to translate to Hindi
def translate_to_hindi(text: str) -> str:
    prompt = f"Translate this sentence to Hindi. Only give translation:\n{text}"
    response = model.generate_content(prompt)
    return response.text.strip()

# Example usage
input_text = "HBLOH0RAreY00"
corrected = correct_sentence(input_text)
print("Corrected:", corrected)

translated = translate_to_hindi(corrected)
print("Hindi Translation:", translated)
