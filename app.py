import gradio as gr
from src.hr_rag_pipeline import HRRagPipeline
from pathlib import Path
import threading
import argparse

pipeline = HRRagPipeline()
is_ready = False


def _parse_args():
    parser = argparse.ArgumentParser(prog='HR Assistant',
                                     description='Answers questions about HR policies.')
    parser.add_argument('--force_reindex', '-f',action='store_true',
                        help='Force reload and re-indexing of HR documents')
    args, _ = parser.parse_known_args()
    return args


args = _parse_args()


def initialize():
    global is_ready
    pipeline.load_and_index(force_reindex=args.force_reindex)
    is_ready = True


threading.Thread(target=initialize, daemon=True).start()

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
    Returns (formatted_answer, chunks_text).
    Passes history to pipeline for multi-turn context resolution.
    """
    result = pipeline.ask_with_history(message, history)

    # Format answer with sources
    sources = set(Path(s).name for s in result["sources"])
    sources_str = ", ".join(sorted(sources))

    answer = f"{result['answer']}\n\n**Sources:** {sources_str}"

    # Format chunks for debug accordion
    chunks_text = ""
    for i, chunk in enumerate(result["contexts"], 1):
        chunks_text += f"**Chunk {i}:**\n{chunk}\n\n---\n"

    return answer, chunks_text


def respond(message: str, history: list) -> tuple[list, str, str]:
    if history is None:
        history = []

    if not message.strip():
        return history, "", ""

    if not is_ready:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant",
                        "content": "Still loading HR documents... please try again in a moment."
                        })
        return history, "", ""
    history.append({"role": "user", "content": message})
    try:
        bot_response, chunks = chat(message, history)
    except Exception as e:
        print(f"Error occurred: {e}")
        history.append({"role": "assistant",
                        "content": "⚠️ Sorry, I encountered an error. Please try again."
                        })
        return history, "", f"**Error:** {str(e)}"

    history.append({"role": "assistant", "content": bot_response})
    return history, "", chunks


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
