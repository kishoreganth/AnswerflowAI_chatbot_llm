CHUNK_SIZE = 400
CHUNK_OVERLAP = 200

VECTOR_DB_PATH = "res/vectordb"


# prompt_template = """Understand the given context with related information and
# answer the question in a polite way. If you don't know the answer, just mention that you don't have information
# in context first and followed with your most relevant answer for the question. Please do not create random answers.



# {context}

# Question: {question} Provide Short and readable strucuted answer for the question given in polite way. Provide answer in original language:"""

CONDENSE_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""

CUSTOM_PROMPT ="""Understand the given information provided and
         answer the question in a polite way. If the question is not related to the provided information or 
         you cant find the answer, provide most relevant answer for the question. 
         Please do not create random answers.

        Context: {context}
        Question: {question} 
        Provide Short and readable strucuted answer for the question given 
        in polite way. Provide answer in original language:
    """
