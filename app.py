import argparse
import os
from pathlib import Path

from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
from smolagents.agent_types import AgentImage
from smolagents.memory import FinalAnswerStep
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from PIL.Image import Image
from Gradio_UI import GradioUI
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DEFAULT_PROMPT = "give me image of current weather in Mogilev"

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def generate_weather_image(location: str) -> Image:
    """Generate an image showing the current weather in a requested location.
    Args:
        location: City or place name, for example 'London' or 'Tokyo'.
    """
    response = requests.get(
        f"https://wttr.in/{location}",
        params={"format": "j1"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    current = data["current_condition"][0]
    description = current["weatherDesc"][0]["value"]
    temp_c = current["temp_C"]
    feels_like_c = current["FeelsLikeC"]
    humidity = current["humidity"]

    prompt = (
        f"A realistic scene in {location} with {description} weather, "
        f"temperature {temp_c}C, feels like {feels_like_c}C, humidity {humidity}%. "
        f"Show the sky, light, clouds, ground conditions, and people/clothing matching this weather. "
        f"High detail, atmospheric, realistic."
    )

    return image_generation_tool(prompt)

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        get_current_time_in_timezone,
        generate_weather_image,
    ],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

def run_agent_prompt(task: str):
    """Run the agent once from the terminal with a prompt."""
    print(f"Running agent with prompt: {task}")
    result = agent.run(task)

    if isinstance(result, FinalAnswerStep):
        result = result.final_answer

    if isinstance(result, AgentImage):
        output_path = Path("generated_weather_mogilev.png")
        result.save(output_path)
        print(f"Image saved to: {output_path.resolve()}")
        return

    if isinstance(result, Image):
        output_path = Path("generated_weather_mogilev.png")
        result.save(output_path)
        print(f"Image saved to: {output_path.resolve()}")
        return

    print("Agent result:")
    print(result)


def main():
    parser = argparse.ArgumentParser(
        description="Run the weather agent in Gradio UI mode or directly from the terminal."
    )
    parser.add_argument(
        "--prompt",
        help="Run the agent once with the provided prompt instead of launching the UI.",
    )
    parser.add_argument(
        "--run-mogilev-weather",
        action="store_true",
        help="Run the agent once with the built-in prompt: 'give me image of current weather in Mogilev'.",
    )
    args = parser.parse_args()

    if args.prompt:
        run_agent_prompt(args.prompt)
        return

    if args.run_mogilev_weather:
        run_agent_prompt(DEFAULT_PROMPT)
        return

    GradioUI(agent).launch()


if __name__ == "__main__":
    main()
