"""Video question answering tool for analyzing video content."""

from typing import Union, Dict, Any
import os
from pathlib import Path
from agents import function_tool
from agents.run_context import RunContextWrapper
from agentz.context.data_store import DataStore
from loguru import logger
import google.generativeai as genai


@function_tool
async def video_qa(
    ctx: RunContextWrapper[DataStore],
    video_url: str,
    question: str
) -> Union[str, Dict[str, Any]]:
    """Asks a question about a video using AI vision capabilities.

    This tool uses Google's Gemini model to analyze video content and answer
    questions about it. The video can be provided as either a local file path
    or a URL.

    Args:
        ctx: Pipeline context wrapper for accessing the data store
        video_url: Path to the video file or URL. Supports local files and HTTP(S) URLs.
        question: The question to ask about the video content.

    Returns:
        String containing the answer to the question about the video,
        or error message if the analysis fails.

    Examples:
        - "What objects are visible in this video?"
        - "Describe the main actions happening in the video"
        - "How many people appear in this video?"
    """
    try:
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY environment variable not set. Please set it to use video_qa."

        genai.configure(api_key=api_key)

        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Prepare the video input
        if video_url.startswith("http://") or video_url.startswith("https://"):
            # Use URL directly
            logger.info(f"Analyzing video from URL: {video_url}")
            video_file = genai.upload_file(path=video_url)
        else:
            # Handle local file path
            video_path = Path(video_url)
            if not video_path.exists():
                return f"Error: Video file not found: {video_url}"

            logger.info(f"Uploading and analyzing local video: {video_path}")
            video_file = genai.upload_file(path=str(video_path.resolve()))

        # Wait for the file to be processed
        logger.info("Waiting for video to be processed...")
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            return f"Error: Video processing failed for {video_url}"

        # Generate response
        logger.info(f"Asking question: {question}")
        response = model.generate_content(
            [video_file, question],
            request_options={"timeout": 600}
        )

        # Extract and return the text response
        answer = response.text
        logger.info(f"Video QA response received: {answer[:100]}...")

        # Store the result in context if needed
        data_store = ctx.context
        if data_store is not None:
            cache_key = f"video_qa:{video_url}:{question}"
            data_store.set(
                cache_key,
                answer,
                data_type="text",
                metadata={
                    "video_url": video_url,
                    "question": question,
                }
            )
            logger.info(f"Cached video QA result with key: {cache_key}")

        return answer

    except Exception as e:
        error_msg = f"Error analyzing video: {str(e)}"
        logger.error(error_msg)
        return error_msg
