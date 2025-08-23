# VLM-Model-fastapi
**VLM-Model-fastapi** is a project that provides a **common API server module for Vision-Language Models (VLM)**.  
It offers a standardized FastAPI-based interface to simplify deployment and integration of various VLM models, featuring:

- **Standardized structure**: Modularized components for model loading, preprocessing, postprocessing, and inference for easy reuse.  
- **Flexible scalability**: Easily add or replace VLM models
- **Production-ready design**: Asynchronous processing, GPU/CPU support, environment-based configuration. 
- **RESTful API**: Endpoints for handling image + text inputs and returning text or multimodal responses.  
- **Cloud/on-premise optimization**: Seamless deployment in Docker and Kubernetes environments.  

## Note
Currently, testing has only been performed with the **VARCO-VISION-2.0-14B** model.  
Further testing with other models will be conducted, and code updates will follow as needed.


## VLM Model Support (To be updated)
- [**VARCO-VISION-2.0-14B**](https://huggingface.co/NCSOFT/VARCO-VISION-2.0-14B)

## Quick Start
- Before starting, you need to download the model listed in [VLM Model Support](#vlm-model-support-to-be-updated).


```bash
# Clone Repository
git clone https://github.com/wklee610/VLM-Model-fastapi.git

# Install requirements.txt
pip install -r requirements.txt

# run
python app/run.py
```

## Environment Configuration

The project uses a `.env.local` file to manage configuration for model serving.
- For **production environments**, it is recommended to use `.env` instead of `.env.local`.
- In this case, update `app/common/env.py`:

```python
# Development
load_dotenv(".env.local")

# Production
load_dotenv(".env")
```





