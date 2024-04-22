import PyPDF2
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.chains import LLMChain, SequentialChain
import os

import streamlit as st

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def question_generator_from_pdf(pdf_path):
    book_text = extract_text_from_pdf(pdf_path)

    llm = Ollama(model="llama2")

    # Summary
    template1 = "Generate the chapter vise summary for revision of given document:\n{book_text}\n"
    prompt1 = ChatPromptTemplate.from_template(template1)
    chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="Summary")

    # MCQs
    template2 = "Generate the chapter vise question from given book:\n{book_text}\n Generate the MCQ type question only"
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="MCQs")

    seq_chain = SequentialChain(chains=[chain1, chain2],
                                input_variables=["book_text"],
                                output_variables=["Summary", "MCQs"],
                                verbose=True)

    return seq_chain(book_text)


if __name__ == "__main__":
    pdf_path = "Maharashtra-board-class-5-EVS-Textbook.pdf"  # Replace with the path to your PDF file
    result = question_generator_from_pdf(pdf_path)
    print(result["Summary"])
    print(result["MCQs"])
