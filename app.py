import gradio as gr
from src.hr_rag_pipeline import HRRagPipeline

# Initialize once at startup
pipeline = HRRagPipeline()
pipeline.load_and_index()

custom_css = """
.chat-container { border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; background: #fafafa; }
.fixed-height-debug { 
    height: 460px; 
    overflow-y: auto; 
    padding: 12px; 
    border-radius: 6px;
}
"""


def chat(message: str, history: list) -> tuple[str, str]:
    """
    Returns (formatted_answer, chunks_text)
    formatted_answer → shown in chat
    chunks_text      → shown in accordion
    """
    result = pipeline.ask(message)

    # Format answer with sources
    sources = set(
        s.split("\\")[-1].split("/")[-1]  # filename only
        for s in result["sources"]
    )
    sources_str = ", ".join(sorted(sources))

    answer = f"{result['answer']}\n\n**Sources:** {sources_str}"

    # Format chunks for debug accordion
    chunks_text = ""
    for i, chunk in enumerate(result["contexts"], 1):
        chunks_text += f"**Chunk {i}:**\n{chunk}\n\n---\n"

    return answer, chunks_text


with gr.Blocks(title="HR Assistant", css=custom_css) as demo:
    gr.Markdown("# 🤖 HR Assistant — ABC Corporation")
    gr.Markdown("Ask me anything about HR policies, leave, and conduct.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=520,
                show_label=False,
                avatar_images=("👤", "🤖")
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask an HR question...",
                    show_label=False,
                    scale=5,
                    container=False
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

        with gr.Column(scale=1, min_width=320):
            with gr.Accordion("🔍 Debug Panel (Click to toggle)", open=True):
                chunks_display = gr.Markdown(
                    value="*Ask a question to see retrieved chunks*",
                    label="Retrieved Chunks",
                    elem_classes="fixed-height-debug"
                )

    # State to store chunks
    chunks_state = gr.State("")

    def respond(message, history):
        if history is None:
            history = []
        if not message.strip():
            return history, "", ""
        history.append({"role": "user", "content": message})

        bot_response, chunks = chat(message, history)
        history.append({"role": "assistant", "content": bot_response})
        # history.append((message, answer))
        return history, "", chunks

    send_btn.click(
        respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, chunks_display]
    )
    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, chunks_display]
    )

    gr.Examples(
        examples=[
            "How many annual leave days do I get?",
            "What is the maternity leave policy?",
            "What is the notice period for resignation?",
            "What happens after a disciplinary violation?"
        ],
        inputs=msg,
        label="Example Questions"
    )

if __name__ == "__main__":
    demo.launch(share=True)
