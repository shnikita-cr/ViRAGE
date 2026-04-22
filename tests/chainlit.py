# app.py
import chainlit as cl
import ollama  # pip install ollama


@cl.on_chat_start
async def start():
    """Optional: send a welcome message when chat starts."""
    await cl.Message(content="Hello! I'm connected to local Ollama. Ask me anything.").send()


@cl.on_message
async def main(message: cl.Message):
    """
    Handle user messages: send to Ollama and stream the response.
    """
    user_input = message.content

    # Create an empty Chainlit message that we'll stream into
    msg = cl.Message(content="")
    await msg.send()

    # Call Ollama's generate with streaming
    stream = ollama.chat(
        model="gemma3:1b",  # or any model you have pulled locally
        messages=[{"role": "user", "content": user_input}],
        stream=True,
    )

    # Accumulate the response and stream token by token
    for chunk in stream:
        if 'message' in chunk and 'content' in chunk['message']:
            token = chunk['message']['content']
            await msg.stream_token(token)

    # Finalize the message
    await msg.update()
