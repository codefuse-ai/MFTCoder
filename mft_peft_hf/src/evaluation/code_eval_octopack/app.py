import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("code_eval_octopack")
launch_gradio_widget(module)
