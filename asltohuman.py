import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
import json  # Add this import for handling JSON data
from fastapi import FastAPI, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ API KEY is not set in the .env file")

def generate_message_from_gestures(gestures):
    try:
        # Construct the prompt based on the gestures
        gestures_str = ", ".join(gestures)
        query = f"Based on the following gestures: {gestures_str}, generate a human-like message that conveys the intended meaning. Only output the message."

        # Create message payload
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": query}]
            }
        ]

        # API Request
        response = requests.post(
            GROQ_API_URL,
            json={"model": "llama-3.2-90b-vision-preview", "messages": messages, "max_tokens": 4000},
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            timeout=30
        )

        # Process response
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return {"error": f"API Error: {response.status_code}"}
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

def convert_asl_to_human_text(asl_sentence):
    try:
        # Construct the prompt for converting ASL to human text
        query = f"Convert the following ASL sentence into grammatically correct human language: '{asl_sentence}'. Only output the message."

        # Create message payload
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": query}]
            }
        ]

        # API Request
        response = requests.post(
            GROQ_API_URL,
            json={"model": "llama-3.2-90b-vision-preview", "messages": messages, "max_tokens": 4000},
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            timeout=30
        )

        # Process response
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return {"error": f"API Error: {response.status_code}"}
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}
