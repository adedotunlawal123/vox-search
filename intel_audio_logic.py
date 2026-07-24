from dotenv import load_dotenv
import voyageai

load_dotenv()

vog_client = voyageai.Client()

from anthropic import Anthropic
import whisper
import re

client = Anthropic()

model = "claude-haiku-4-5"