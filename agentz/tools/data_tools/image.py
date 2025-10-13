"""Image analysis and question answering tools for processing visual content."""

from typing import Union, Dict, Any, Optional
import os
import base64
from io import BytesIO
from pathlib import Path
from agents import function_tool
from agents.run_context import RunContextWrapper
from agentz.context.data_store import DataStore
from loguru import logger
import google.generativeai as genai
from PIL import Image
import requests


def _load_image(image_path: str) -> Image.Image:
    """Load an image from a local path or URL.

    Args:
        image_path: Path to the image file or URL

    Returns:
        PIL Image object

    Raises:
        Exception: If image cannot be loaded
    """
    try:
        if image_path.startswith("http://") or image_path.startswith("https://"):
            # Load from URL
            logger.info(f"Fetching image from URL: {image_path}")
            response = requests.get(image_path, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            # Load from local file
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            logger.info(f"Loading local image: {image_path}")
            image = Image.open(image_file)

        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        return image

    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise


def _image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to base64 encoded string.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded string of the image
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


@function_tool
async def image_qa(
    ctx: RunContextWrapper[DataStore],
    image_path: str,
    question: Optional[str] = None
) -> Union[str, Dict[str, Any]]:
    """Analyzes an image and answers questions about it using AI vision capabilities.

    This tool uses Google's Gemini model to analyze image content. If no question
    is provided, it generates a detailed description of the image. If a question
    is provided, it answers the specific question about the image.

    Args:
        ctx: Pipeline context wrapper for accessing the data store
        image_path: Path to the image file or URL. Supports local files and HTTP(S) URLs.
        question: Optional question to ask about the image. If None, generates a
                 detailed description of the image.

    Returns:
        String containing either the image description or the answer to the question,
        or error message if the analysis fails.

    Examples:
        - image_qa(ctx, "photo.jpg") -> Generates detailed description
        - image_qa(ctx, "photo.jpg", "What color is the car?") -> Answers specific question
        - image_qa(ctx, "https://example.com/image.jpg", "How many people are in this image?")
    """
    try:
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY environment variable not set. Please set it to use image_qa."

        genai.configure(api_key=api_key)

        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Load the image
        logger.info(f"Loading image from: {image_path}")
        image = _load_image(image_path)

        # Prepare the prompt
        if question is None:
            prompt = "Provide a detailed description of this image, including objects, people, actions, colors, and any text visible."
            logger.info("Generating detailed image description")
        else:
            prompt = question
            logger.info(f"Asking question: {question}")

        # Generate response
        response = model.generate_content([prompt, image])

        # Extract and return the text response
        answer = response.text
        logger.info(f"Image QA response received: {answer[:100]}...")

        # Store the result in context if needed
        data_store = ctx.context
        if data_store is not None:
            cache_key = f"image_qa:{image_path}:{question or 'description'}"
            data_store.set(
                cache_key,
                answer,
                data_type="text",
                metadata={
                    "image_path": image_path,
                    "question": question,
                    "image_size": image.size,
                }
            )
            logger.info(f"Cached image QA result with key: {cache_key}")

        return answer

    except FileNotFoundError as e:
        error_msg = str(e)
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        logger.error(error_msg)
        return error_msg
