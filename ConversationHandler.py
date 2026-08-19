class ConversationHandler:

    def __init__(self):
        self.messages = []

    def user_prompt_adder(self, question: str, needed_context: str):
        user_prompt = {
            "role": "user",
            "content": f"""Your task is to answer the question, 
            based on the context below. Use the context to find relevant information. 
            If the answer is not found in the context, say "I dont have the information to answer that question".

The Question is
<question>
{question}



The context is 
<context>
{needed_context}
<\context>
"""
}
self.messages.append(user_prompt)



def assistant_message_adder(self, text:str):
    assistant_message = {"role":"assistant", "content":text}
    self.messages.append(assistant_message)

def take_input():
    user_input = input("Type 'exit' to stop: ")
    print(user_input)
    return user_input

def context(user_input: str):
    results = retriever.search(f"{user_input}")
    needed_context = results[0][0]["content"]
    return needed_context


def ping_LLM(messages: list):
    message = client.messages.create(
        model=model,
        max_tokens=500,
        messages=messages,
  )
    return message.content[0].text


def converse_with_LLM():
    user_input = self.take_input()
    needed_context = self.context(user_input)
    #Add user input
    self.user_prompt_adder(self.messages, user_input, needed_context)
    #Get Claude's response
    ai_answer=self.ping_LLM(self.messages)
    #Add response from claude (assistant) to my message using the adder function
    self.assistant_message_adder(self.messages, ai_answer)
    print(ai_answer)