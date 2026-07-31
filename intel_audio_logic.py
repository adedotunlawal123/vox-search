from dotenv import load_dotenv
import voyageai
load_dotenv()
vog_client = voyageai.Client()
from anthropic import Anthropic
#import whisper
import re
client = Anthropic()
model = "claude-haiku-4-5"
from ChunckAndEmbed import chunk_by_sentence as chunker
from ChunckAndEmbed import generate_embedding as embedder
from HybridSearchImplementation import VectorIndex
from HybridSearchImplementation import BM25Index
from HybridSearchImplementation import Retriever




# Load the model
#trnscrb_model = whisper.load_model("base")
#Transcribe the audio
#result = trnscrb_model.transcribe(r"C:\Users\adedo\Downloads\What are Transformers (Machine Learning Model)_.mp3")
# Print the text
#print(result["text"])
#write the transcript into  file
#with open("Learn_Transformer", "w", encoding="utf-8") as file:
    #file.write(result["text"])



# Chunk source text by section
with open("Learn_Transformer", "r") as f:
    text = f.read()

#chunk the text
chunks = chunker(text)

#Generate a vector embedding of the chunck using VogayeAI; See HybridSearchImplementation package for more details
vector_index = VectorIndex(embedding_fn=embedder)
#Instantiate the BestMatch 25 class for key word search
bm25_index = BM25Index()

#Instantiate class thatdoes the hybrid search See HybridSearchImplementation package for more details
retriever = Retriever(bm25_index, vector_index)

# Add all chunks to the retriever, which internally passes them along to both indexes
retriever.add_documents([{"content": chunk} for chunk in chunks])

message = []


def user_prompt_adder(message, question, needed_context):
    user_prompt = { "role":"user",
                    "content": f"""Your task is to answers question, 
                    based on the context below. Use the context to find relevant information. 
                    If the answer is not found in the context, say "I dont Know".

The Question is
<question>
{question}



The context is 
<context>
{needed_context}
<\context>
"""
}
    
    message.append(user_prompt)



def assistant_message_adder(message, text):
    assistant_message = {"role":"assistant", "content":text}
    message.append(assistant_message)

def take_input():
    user_input = input("Type 'exit' to stop: ")
    print(user_input)
    return user_input

def context(user_input: str):
    results = retriever.search(f"{user_input}")
    needed_context = results[0][0]["content"]
    return needed_context

def ping_LLM(messages):
    message = client.messages.create(
        model=model,
        max_tokens=500,
        messages=messages,
    )
    return message.content[0].text

def converse_with_LLM():

    user_input = take_input()

    needed_context = context(user_input)
    
    #Add user input
    user_prompt_adder(message, user_input, needed_context)

    #Get Claude's response
    ai_answer=ping_LLM(message)

    #Add response from claude (assistant) to my message using the adder function
    assistant_message_adder(message, ai_answer)

    print(ai_answer)


