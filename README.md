# VLM-Model-fastapi
**VLM-Model-fastapi** is a project that provides a **common API server module for Vision-Language Models (VLM)**.  
It offers a standardized FastAPI-based interface to simplify deployment and integration of various VLM models, featuring:

- **Standardized structure**: Modularized components for model loading, preprocessing, postprocessing, and inference for easy reuse.  
- **Flexible scalability**: Easily add or replace VLM models
- **Production-ready design**: Asynchronous processing, GPU/CPU support, environment-based configuration. 
- **RESTful API**: Endpoints for handling image + text inputs and returning text or multimodal responses.  
- **Cloud/on-premise optimization**: Seamless deployment in Docker and Kubernetes environments.  

## Note
- Currently, testing has only been performed with the **VARCO-VISION-2.0-14B** model.  
- Further testing with other models will be conducted, and code updates will follow as needed.
- Check `.env.varco` first to run.

## VLM Model Support (To be updated)
- [**VARCO-VISION-2.0-14B**](https://huggingface.co/NCSOFT/VARCO-VISION-2.0-14B)

## Quick Start
- Before starting, you need to download the model listed in [VLM Model Support](#vlm-model-support-to-be-updated).


```bash
# Clone Repository
git clone https://github.com/wklee610/VLM-Model-fastapi.git

# Virtual Env
python -m venv .venv

# Install requirements.txt
pip install -r requirements.txt

# Fix .env.varco or make new .env

# Run
python app/run.py

# Test
curl -X POST "http://localhost:8080/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '[
           {
             "role": "user",
             "content": [
               {
                 "type": "text",
                 "text": "안녕하세요?"
               }
             ]
           }
         ]'
```

## Environment Configuration

The project uses a `.env.varco` file to manage configuration for model serving.
- For **production environments**, it is recommended to use `.env` instead of `.env.varco`.
- In this case, update `app/common/env.py`:

```python
# Development
load_dotenv(".env.varco")

# Production
load_dotenv(".env")
```

## TODO
- print -> logging
- doc string
- gpu manager change (better one)
- exception / error handler

