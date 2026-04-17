import gradio as gr

from backend_tagger.runtime import predictor, utils


TITLE = "WaifuDiffusion Tagger"
DESCRIPTION = """
<br>
- API docs: <code>/docs</code> or <code>/redoc</code><br>
- API endpoint: <code>/tagger/v1</code>
"""


def reset_form():
    model_names = utils.refresh_interrogators()
    default_model = "wd14-eva02-large-v3-git" if "wd14-eva02-large-v3-git" in model_names else (model_names[0] if model_names else None)
    return [None, default_model, 0.35, False, 0.85, False] + [None] * 8


def refresh_model_dropdown(current_model: str | None):
    model_names = utils.refresh_interrogators()
    if current_model in model_names:
        selected_model = current_model
    else:
        selected_model = "wd14-eva02-large-v3-git" if "wd14-eva02-large-v3-git" in model_names else (model_names[0] if model_names else None)
    return gr.update(choices=model_names, value=selected_model)


def create_ui() -> gr.Blocks:
    with gr.Blocks(title=TITLE, analytics_enabled=False) as demo:
        gr.Markdown(f"<h1 style='text-align: center'>{TITLE}</h1>")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(variant="panel"):
                image = gr.Image(type="pil", image_mode="RGBA", label="Input Image")
                with gr.Row():
                    model = gr.Dropdown(choices=utils.all_model_names, label="Model", value="wd14-eva02-large-v3-git", scale=8)
                    refresh_models = gr.Button("Refresh", scale=1, min_width=90)
                with gr.Row():
                    general_thresh = gr.Slider(0, 1, 0.35, 0.05, label="General Threshold")
                    general_mcut = gr.Checkbox(False, label="Use MCut for general tags")
                with gr.Row():
                    char_thresh = gr.Slider(0, 1, 0.85, 0.05, label="Character Threshold")
                    char_mcut = gr.Checkbox(False, label="Use MCut for character tags")
                with gr.Row():
                    clear = gr.ClearButton(variant="secondary")
                    submit = gr.Button("Interrogate", variant="primary")

            with gr.Column(variant="panel"):
                output_str = gr.Textbox(label="Prompt Tags", lines=5)
                with gr.Row():
                    ratings_out = gr.Label(label="Ratings", num_top_classes=4)
                    quality_out = gr.Label(label="Quality")
                with gr.Row():
                    artist_out = gr.Label(label="Artist")
                    copyright_out = gr.Label(label="Copyright")
                characters_out = gr.Label(label="Characters", num_top_classes=5)
                with gr.Row():
                    meta_out = gr.Label(label="Meta")
                    general_out = gr.Label(label="General Tags")

        inputs = [image, model, general_thresh, general_mcut, char_thresh, char_mcut]
        outputs = [output_str, ratings_out, quality_out, characters_out, artist_out, copyright_out, meta_out, general_out]

        clear.click(fn=reset_form, outputs=inputs + outputs, show_progress=False)
        refresh_models.click(fn=refresh_model_dropdown, inputs=[model], outputs=[model], show_progress=False)
        submit.click(fn=predictor.predict, inputs=inputs, outputs=outputs)

    return demo


def on_ui_tabs():
    block = create_ui()
    return [(block, "WD14 Tagger", "wd14_tagger")]
