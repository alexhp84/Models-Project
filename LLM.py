import os
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import logging

# Set up a module‑level logger
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("[%(levelname)s] %(asctime)s – %(name)s – %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Load environment variables from .env
load_dotenv()

#Parameters from the .env
try:
    API_KEY = os.environ["LLM_API_KEY"]
    ENDPOINT = os.environ["LLM_ENDPOINT"]
    MODEL = os.environ["LLM_MODEL"]
except KeyError as missing_key:
    raise RuntimeError(
        f"Missing required .env variable: {missing_key}. "
        "Set LLM_API_KEY, LLM_ENDPOINT, and LLM_MODEL before running."
    ) from missing_key

#Create an LLM session with retries
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
)
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))

def ask_llm(prompt: str) -> str:
    """
    Send a user prompt to the LLM endpoint and return the generated reply.
    """
    logger.info(f"Calling LLM endpoint {ENDPOINT} with model {MODEL}")

    try:
        response = session.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("choices") or not isinstance(data["choices"], list):
            raise ValueError("Unexpected LLM response structure")

        return data["choices"][0]["message"]["content"]

    except requests.HTTPError as http_err:
        logger.error(f"HTTP error during LLM call: {http_err}")
        raise
    except Exception as exc:
        logger.exception(f"Unexpected error during LLM call: {exc}")
        raise