import gradio as gr
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from src.rag_lanchain import graph_builder


memory = MemorySaver()
in_memory_store = InMemoryStore()
graph = graph_builder.compile(checkpointer=memory, store=in_memory_store)


def respond(msg, config):
    role_dict = {"ai": "assistant", "human": "user"}
    if len(msg) == 0:
        gr.Warning("Chat messages cannot be empty")
        history = []
        for hist in graph.get_state_history(config):
            history = [{"role": role_dict.get(i.type, i.type), "content": i.content} for i in
                       hist.values["messages"]]
            break
        return "", history
    events = graph.stream(
        {"messages": [{"role": "user", "content": msg}]},
        config,
        stream_mode="values",
    )
    events = list(events)
    conversation = events[-1]["messages"]
    conversation = [{"role": role_dict.get(i.type, i.type), "content": i.content} for i in conversation]
    return "", conversation


def init_chat_state():
    return {"configurable": {"thread_id": str(uuid4()).replace('-', '_')}}


css = """
.centered-container {
    max-width: 1000px;
    margin: 0 auto;
}
"""

THEME = gr.themes.Ocean()

demo = gr.Blocks(theme=THEME, fill_width=False, fill_height=True, css=css)

with demo:
    config_state = gr.State(init_chat_state)
    with gr.Column(elem_classes="centered-container"):
        gr.Markdown("""
        # 💬 Polars Python Chatbot
        ### Ask anything about the [Polars](https://pola-rs.github.io/polars/) Python package!  
        ### This chatbot uses a database of embeddings generated from the official documentation to help you find accurate and relevant answers about using Polars for data manipulation in Python.
        """)

        chatbot = gr.Chatbot(
            label=None,
            type="messages",
            show_label=False,
            height=400,
        )

        with gr.Row(equal_height=True):
            msg = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                lines=3,
                max_lines=3,
                scale=5,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear = gr.ClearButton([msg, chatbot], value="Clear Chat", variant="secondary")

    send_btn.click(respond, [msg, config_state], [msg, chatbot])
    msg.submit(respond, [msg, config_state], [msg, chatbot])

if __name__ == '__main__':
    demo.launch()
