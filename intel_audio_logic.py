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
from ConversationHandler import ConversationHandler, converse_with_LLM




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


#Instantiate the ConversationHandler class
convo = ConversationHandler()

#Start the conversation with the LLM
convo.converse_with_LLM()
