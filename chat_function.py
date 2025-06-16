import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
import base64
import os 
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.document import Document
import time
from pypdf import PdfReader
import re
import copy
import io
import pdfplumber
from PIL import Image

def clean_text(text):
    # Replace more than 2 consecutive newlines with exactly 2 newlines
    text = re.sub(r'\n{1,}', '\n', text)
    return text.strip()

def extract_text_and_image_summary(file_stream, extracted_text_file_name, image_folder, gemini_model):
    reader = PdfReader(file_stream)

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(extracted_text_file_name):
        extract_text_lst = []

        page_image_map = {}  # Map page number to list of image file paths

        # Step 1: Extract text and save images with page number
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_num = i + 1
            saved_images = []

            for image in page.images:
                name, ext = os.path.splitext(image.name)
                new_image_name = f"{name}_page{page_num}{ext}"
                image_path = os.path.join(image_folder, new_image_name)

                image_path = os.path.join(image_folder, new_image_name)
                with open(image_path, 'wb') as f:
                    f.write(image.data)
                # Load the image

                image = Image.open(image_path)  # Replace with your image path

                # Get width and height
                width, height = image.size
                print('width:',width,'hight:',height)
                # Check if both dimensions are greater than 90
                if (width == 90 and height == 90) or (width == 214 and height == 115) or  (width == 1655 and height == 279):
                    print(i)
                    print(f"{width,height} is larger than 90x90")
                    saved_images.append('')
                else:
                    if not os.path.exists(image_path):
                        with open(image_path, 'wb') as f:
                            f.write(image.data)
                        print(f"Saved: {new_image_name}")
                    else:
                        print(f"File already exists: {new_image_name}")

                    saved_images.append(image_path.replace("\\", "/"))

            page_image_map[page_num] = saved_images
            extract_text_lst.append({"text": page_text, "images": saved_images})
        # Step 2: Process images and generate summaries
        for page_index, entry in enumerate(extract_text_lst):

            image_summaries = []
            print('entry_image_len:',len(entry["images"]))
            for image_path in entry["images"]:
                if image_path == '':
                    continue
                else:
                    ext = image_path.split('.')[-1]

                    with open(image_path, "rb") as img_file:
                        b64_string = base64.b64encode(img_file.read()).decode('utf-8')

                    prompt_template = """You are an assistant. Most images show steps to perform tasks. \
                                        Provide a concise, max five-line step summary based only on the image. Do not invent or add information."""

                    messages = [
                        (
                            "user",
                            [
                                {"type": "text", "text": prompt_template},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/{ext};base64,{b64_string}"},
                                },
                            ],
                        )
                    ]

                    try:
                        prompt = ChatPromptTemplate.from_messages(messages)
                        chain = prompt | gemini_model | StrOutputParser()
                        result = chain.invoke({"b64_img": b64_string, "ext": ext})
                        print('added.........sss:',page_index)
                        image_summaries.append(result)
                        time.sleep(3)  # respect rate limits
                        
                    except Exception as e:
                        print(f"Error: {e}")
                        image_summaries.append("Error processing image.")

            print('::::::::::::::::',image_summaries)
            # Combine text and image summary
            page_text = entry["text"]
            if image_summaries:
                page_text += "\n\n Example related to Topic:\n" + "\n".join(image_summaries)

            # Replace dict with combined text
            extract_text_lst[page_index] = page_text

        # Step 3: Write to output file
        final_text = "\n\n--- PAGE BREAK ---\n\n".join(extract_text_lst)
        with open(extracted_text_file_name, "w") as f:
            f.write(final_text)

    return 0


def get_prompt():
    prompt = '''You are a helpful assistant. Provide answers **only** from the data provided.  
    If the answer is not found in the provided data, do **not** generate your own answer.  
    If the question is unclear or confusing, ask follow-up questions to clarify before answering.

    **Note**: if any user ask that information from pdf , document it means what content provided to you 
    from that you have to answer.
    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}
    '''
    
    
    # create prompt 
    prompt = PromptTemplate(template=prompt,input_variables=['context','chat_history','question'])
    
    return prompt


def split_text(data):

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=700,
        chunk_overlap=100,
        length_function=len
    )

    chunk_text = text_splitter.split_text(data)

    return chunk_text


def get_chain(llm,prompt,retriever,memory):

    chain = ConversationalRetrievalChain.from_llm(
                                                llm=llm, retriever=retriever,
                                                memory = memory,
                                                return_source_documents=False,
                                                combine_docs_chain_kwargs={'prompt': prompt})

    return chain



