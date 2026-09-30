from anthropic import Anthropic

DEFAULT_MODEL = "claude-haiku-4-5"


class ConversationHandler:

    def __init__(self, retriever, client=None, model=DEFAULT_MODEL):
        self.messages = []
        # The retriever and the Anthropic client are passed in rather than pulled
        # from module globals, so this class stays independent of the entry point.
        self.retriever = retriever
        self.client = client or Anthropic()
        self.model = model

    def user_prompt_adder(self, question: str, needed_context: str):
        user_prompt = {
            "role": "user",
            "content": f"""Your task is to answer the question, 
            based on the context below. Use the context to find relevant information. 
            If the answer is not found in the context, say "I dont have the information to answer that question".

        The Question is
        <question>
        {question}
        </question>



        The context is 
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
        results = self.retriever.search(f"{user_input}")
        if not results:
            return "No relevant context was found in the transcript."
        needed_context = results[0][0]["content"]
        return needed_context


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

            needed_context = self.context(user_input)
            #Add user input
            self.user_prompt_adder(user_input, needed_context)
            #Get Claude's response
            ai_answer = self.ping_LLM(self.messages)
            #Add response from claude (assistant) to my message using the adder function
            self.assistant_message_adder(ai_answer)
            print(f"\n{ai_answer}")
