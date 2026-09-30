from anthropic import Anthropic

from transcribe import format_clock

DEFAULT_MODEL = "claude-haiku-4-5"


class ConversationHandler:

    def __init__(self, retriever, client=None, model=DEFAULT_MODEL):
        self.messages = []
        # The retriever and the Anthropic client are passed in rather than pulled
        # from module globals, so this class stays independent of the entry point.
        self.retriever = retriever
        self.client = client or Anthropic()
        self.model = model

    def user_prompt_adder(self, question: str, needed_context: str, timestamp: str = ""):
        # The timestamp says where in the audio this context was spoken, so Claude
        # can tell the user when a statement was made.
        when = f' spoken at {timestamp} in the audio' if timestamp else ""
        cite = (
            f'\n        When you use the context, mention that it was said at {timestamp}.'
            if timestamp else ""
        )
        user_prompt = {
            "role": "user",
            "content": f"""Your task is to answer the question, 
            based on the context below. Use the context to find relevant information. 
            If the answer is not found in the context, say "I dont have the information to answer that question".{cite}

        The Question is
        <question>
        {question}
        </question>



        The context is{when} 
        <context>
        {needed_context}
        </context>
        """
        }
        self.messages.append(user_prompt)



    def assistant_message_adder(self, text:str):
        assistant_message = {"role":"assistant", "content":text}
        self.messages.append(assistant_message)

    def take_input(self):
        return input("\nAsk about the audio (or 'exit' to quit): ").strip()

    def context(self, user_input: str):
        """Best-matching chunk as a dict, so its timestamps travel with the text."""
        results = self.retriever.search(f"{user_input}")
        if not results:
            return {"content": "No relevant context was found in the transcript."}
        return results[0][0]

    @staticmethod
    def timestamp_of(document: dict) -> str:
        """Render a chunk's position in the audio, e.g. 12:34-13:01.

        Empty when the transcript carried no timing (a plain-text transcript).
        """
        if document.get("start") is None or document.get("end") is None:
            return ""
        return f"{format_clock(document['start'])}-{format_clock(document['end'])}"


    def ping_LLM(self, messages: list):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=messages,
    )
        return message.content[0].text


    def converse_with_LLM(self):
        """Keep answering questions until the user types 'exit'."""
        while True:
            user_input = self.take_input()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            document = self.context(user_input)
            timestamp = self.timestamp_of(document)
            #Add user input
            self.user_prompt_adder(user_input, document["content"], timestamp)
            #Get Claude's response
            ai_answer = self.ping_LLM(self.messages)
            #Add response from claude (assistant) to my message using the adder function
            self.assistant_message_adder(ai_answer)
            print(f"\n{ai_answer}")
            if timestamp:
                print(f"\n  [audio {timestamp}]")
