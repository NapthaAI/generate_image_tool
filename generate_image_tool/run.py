import os
import base64
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import Dict
from generate_image_tool.schemas import InputSchema
from naptha_sdk.schemas import ToolDeployment, ToolRunInput
from naptha_sdk.user import sign_consumer_id

logger = logging.getLogger(__name__)
load_dotenv()

class GenerateImageTool():
    def __init__(self, tool_deployment: ToolDeployment):
        self.tool_deployment = tool_deployment
        self.api_key = os.environ['STABILITY_API_KEY']
        if self.api_key is None:
            raise ValueError("API key is not set")

    def generate_image_tool(self, inputs: InputSchema):
        """Run the module to generate image from text prompt using Stability API"""
        logger.info(f"Generating image from prompt: {inputs.prompt}")
        
        default_engine = self.tool_deployment.config.llm_config.model
        default_filename = self.tool_deployment.data_generation_config.default_filename
        
        # Handle text-to-image generation
        url = f"{self.tool_deployment.config.llm_config.api_base}/v1/generation/{default_engine}/text-to-image"
        
        data = {
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 1024,
            "width": 1024,
            "sampler": "K_DPM_2_ANCESTRAL",
            "samples": 1,
            "steps": 30,
            "text_prompts": [
                {
                    "text": inputs.prompt,
                    "weight": 1
                }
            ]
        }

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}"
            },
            json=data
        )

        return self._handle_response(response, default_filename)

    def image_to_image_tool(self, inputs: InputSchema):
        """Run the module to generate image from input image and text prompt using Stability API"""
        logger.info(f"Generating image from prompt and input image: {inputs.prompt}")
        
        default_engine = self.tool_deployment.config.llm_config.model
        default_filename = self.tool_deployment.data_generation_config.default_filename
        url = f"{self.tool_deployment.config.llm_config.api_base}/v1/generation/{default_engine}/image-to-image"

        # Handle input image
        try:
            image_path = Path(inputs.input_dir).glob('*').__next__()
            image = Image.open(image_path)
            image = image.resize((1024, 1024))
            temp_path = "/tmp/init_image.png"
            image.save(temp_path)
            files = {
                "init_image": open(temp_path, "rb")
            }
        except Exception as e:
            raise ValueError("No image provided. Must provide either input_dir or base64 image")

        response = requests.post(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            files=files,
            data={
                "image_strength": 0.35,
                "init_image_mode": "IMAGE_STRENGTH",
                "text_prompts[0][text]": inputs.prompt,
                "cfg_scale": 7,
                "samples": 1,
                "steps": 30,
            }
        )

        return self._handle_response(response, default_filename)

    def _handle_response(self, response, default_filename):
        """Handle API response and save image if required"""
        if response.status_code != 200:
            logger.error(f"Failed to generate image: {response.text}")
            raise ValueError(f"Failed to generate image: {response.text}")

        result = response.json()
        image_b64 = result['artifacts'][0]['base64']
        image = Image.open(BytesIO(base64.b64decode(image_b64)))

        if self.tool_deployment.data_generation_config.save_outputs_path:
            output_path = self.tool_deployment.data_generation_config.save_outputs_path
            Path(output_path).mkdir(parents=True, exist_ok=True)
            image.save(f"{output_path}/{default_filename}")
            return f"Image saved to {output_path}/{default_filename}"
        return "Image generated successfully"

def run(module_run: Dict, *args, **kwargs):
    """Run the module to generate image using Stability API"""
    module_run = ToolRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    generate_image_tool = GenerateImageTool(module_run.deployment)
    method = getattr(generate_image_tool, module_run.inputs.tool_name, None)
    if not method:
        raise ValueError(f"Method {module_run.inputs.tool_name} not found")
    result = method(module_run.inputs)
    return result

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    naptha = Naptha()
    deployment = asyncio.run(setup_module_deployment("tool", "generate_image_tool/configs/deployment.json", node_url=os.getenv("NODE_URL")))

    # Example 1: Text to Image
    input_params_text = {
        "tool_name": "generate_image_tool",
        "tool_input_data": "expansive landscape rolling greens with gargantuan yggdrasil, intricate world-spanning roots towering under a blue alien sky, masterful, ghibli"
    }

    # Example 2: Image to Image with input_dir
    input_params_image_dir = {
        "tool_name": "image_to_image_tool",
        "tool_input_data": {
            "prompt": "A beautiful sunset over the ocean",
            "input_dir": "./input_folder"
        }
    }
    # Choose which example to run
    module_run = {
        "inputs": input_params_text,  # or input_params_image_dir or input_params_image_base64
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = run(module_run)
    print("Response: ", response)