import sys
sys.path.append('./')

import os
import gc
import random
import argparse
import traceback
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
from typing import Union, Literal, List, Tuple

import PIL
from PIL import Image
from io import BytesIO

import torch
import numpy as np

import uvicorn

from fastapi import FastAPI, Form, File, UploadFile, Request, status
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse, Response
from fastapi.exceptions import HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware

from src.api.templates import OutputAPI

from src.utils.io_util import load_config, load_multipart_file
from src.utils.dtype_util import image2base64
from src.utils.profiler_util import get_gpu_memory, get_gpu_profile, get_cpu_info


#############################
#       Model Config        #
#############################

MODEL_CONFIG = dict(

    # GPU Settings
    cpu_offload = False,
    cuda_device = -1,

    # Checkpoints Directory
    MODEL_DIR = os.environ.get('MODEL_DIR', './'),

    # Model checkpoints - Human
    humodel_mtf_path = "{MODEL_DIR}/mtf/{model}/model{model_ep:09}.pt",
    humodel_m2m_path = "{MODEL_DIR}/m2m/{model}/model{model_ep:09}.pt",
    humodel_a2m_path = "{MODEL_DIR}/a2m/{model}/model{model_ep:09}.pt",
    humodel_t2m_path = "{MODEL_DIR}/t2m/{model}/model{model_ep:09}.pt",
    humodel_t2mm_path = "{MODEL_DIR}/lm/{model}/model{model_ep:09}.pt",
    humodel_t2cm_path = "{MODEL_DIR}/cm/{model}/model{model_ep:09}.pt",
    humodel_ctrl_path = "{MODEL_DIR}/mc/{model}/model{model_ep:09}.pt",

    humodel_mtf_opt = { 'momo': 500_000},
    humodel_a2m_opt = { 'humanact12': 350_000, 'humanact12_no_fc':   750_000,
                             'uestc': 950_000,      'uestc_no_fc': 1_550_000, },
    humodel_t2m_opt = {    'humanml_trans_dec_512': 375_000,
                           'humanml_trans_enc_512': 475_000, },
    humodel_t2mm_opt = {'my_humanml_trans_enc_512': 200_000,  
                          'Babel_TrasnEmb_GeoLoss': 850_000, },
    humodel_t2cm_opt = {'pw3d_prefix' : 240_000, 
                        'pw3d_text'   : 100_000, },
    humodel_ctrl_opt = {'left_wrist'  : 280_000,
                        'right_foot'  : 280_000,
                        'root_horizon': 280_000, },
    humodel_m2m_opt = { 'mixamo/0000_Breakdance_Freezes'    : 60_000,
                        'mixamo/0001_Capoeira'              : 60_000,
                        'mixamo/0002_Chicken_Dance'         : 60_000,
                        'mixamo/0003_Dancing'               : 60_000,
                        'mixamo/0004_House_Dancing'         : 60_000,
                        'mixamo/0005_Punch_To_Elbow_Combo'  : 60_000,
                        'mixamo/0006_Salsa_Dancing'         : 60_000,
                        'mixamo/0007_Swing_Dancing'         : 60_000,
                        'mixamo/0008_Warming_Up'            : 60_000,
                        'mixamo/0009_Wave_Hip_Hop_Dance'    : 60_000, },

    # TODO: Model checkpoints - Human-Object Interaction

    # Model checkpoints - Animal
    animodel_m2m_path = "{MODEL_DIR}/m2m/animals/{model}/model{model_ep:09}.pt",
    animodel_m2m_opt = {  'horse':  39_999,  'jaguar': 130_000,
                        'ostrich': 140_000, 'weavern': None, },
)


#############################
#         API Setup         #
#############################

API_CONFIG = load_config(path='./src/api/config.yaml')
API_RESPONDER = OutputAPI()

app = FastAPI(
      root_path =  os.getenv("ROOT_PATH"), 
          title = API_CONFIG['DESCRIPTION']['title'],
    description = API_CONFIG['DESCRIPTION']['overview'],
   openapi_tags = API_CONFIG['TAGS'],
        version = API_CONFIG['VERSION'],
       docs_url = None, 
      redoc_url = None,
)

app.add_middleware(
    CORSMiddleware,         # https://fastapi.tiangolo.com/tutorial/cors/
    allow_origins=['*'],    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
    allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
    allow_methods=['*'],    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
    allow_headers=['*'],    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
)


#############################
#       API Endpoints       #
#############################

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
    query_params = str(request.query_params)
    openapi_url = app.root_path + app.openapi_url + "?" + query_params
    return get_swagger_ui_html(
        openapi_url = openapi_url,
                title = "Hari Yu - Demo",
    swagger_favicon_url = "https://github.com/HariWu1995/InstantID-FaceSwap/blob/main/assets/favicon.ico",
    )


@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.post("/models/config")
async def models_config(
          MODEL_DIR :  str = Form(description=API_CONFIG['PARAMETERS'][ 'MODEL_DIR' ], default=MODEL_CONFIG['MODEL_DIR']), 
        cuda_device :  int = Form(description=API_CONFIG['PARAMETERS']['cuda_device'], default=MODEL_CONFIG['cuda_device']), 
        cpu_offload : bool = Form(description=API_CONFIG['PARAMETERS']['cpu_offload'], default=MODEL_CONFIG['cpu_offload']), 
):
    try:
        global MODEL_CONFIG
        MODEL_CONFIG.update(dict(
                 MODEL_DIR = MODEL_DIR,
               cuda_device = cuda_device,
               cpu_offload = cpu_offload,
        ))
        response = API_RESPONDER.result(is_successful=True, data=MODEL_CONFIG)

    except Exception as e:
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())
    
    return response


@app.post("/generate")
async def generate(
          face_image: UploadFile = \
                            File(description=API_CONFIG['PARAMETERS']['face_image'], media_type='multipart/form-data'),
          pose_image: UploadFile = \
                            File(description=API_CONFIG['PARAMETERS']['pose_image'], media_type='multipart/form-data'),
      mask_strength : float = Form(description=API_CONFIG['PARAMETERS']['mask_strength'], default=0.99), 
      mask_padding_W:   int = Form(description=API_CONFIG['PARAMETERS']['mask_padding_W'], default=19), 
      mask_padding_H:   int = Form(description=API_CONFIG['PARAMETERS']['mask_padding_H'], default=11), 
      mask_threshold: float = Form(description=API_CONFIG['PARAMETERS']['mask_threshold'], default=0.33), 
             prompt: str = Form(description=API_CONFIG['PARAMETERS']['prompt_positive'], default=PROMPT_POSITIVE), 
):

    try:        
        # Preprocess
        filename = face_image.filename
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in [".png", ".jpg", ".jpeg"]:
            raise TypeError(f"{filename} with type")

        face_image = await face_image.read()
        face_image = Image.open(BytesIO(face_image)).convert('RGB')

        pose_image = await pose_image.read()
        pose_image = Image.open(BytesIO(pose_image)).convert('RGB')

        # Run pipeline
        generated_image = swap_face_only()

        # Response
        print('\nResponding ...')
        if return_output_only:
            if isinstance(generated_image, np.ndarray):
                image_in_bytes = generated_image.tobytes()
            elif isinstance(generated_image, PIL.Image.Image):
                image_in_bytes = BytesIO()
                generated_image.save(image_in_bytes, format='PNG')
                image_in_bytes = image_in_bytes.getvalue()
            else:
                raise TypeError(f"Type of output = {generated_image.__class__} is not supported!")

            response = Response(content=image_in_bytes, media_type="image/png")
            # response = API_RESPONDER.result(is_successful=True, data=results)
        
        else:
            if isinstance(generated_image, np.ndarray):
                generated_image = Image.fromarray(generated_image.astype(np.uint8))
            generated_image.save('logs/output.png')

            images_fn = ['bbox.png','segment.png','contour.png','mask.png','output.png',
                         'portrait.png','portrait_mask.png','portrait_kpts.png']
                
            buffer = BytesIO()
            archive = ZipFile(buffer, mode='w', compression=ZIP_DEFLATED)
            # archive.setpassword(b"secret")
            for fname in images_fn:
                archive.write('logs/'+fname)
            archive.close()

            return StreamingResponse(
                iter([buffer.getvalue()]), 
                media_type="application/x-zip-compressed", 
                headers={"Content-Disposition": "attachment; filename=images.zip"}
            )

    except Exception as e:
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())

    return response


@app.post("/clear")
async def clear():
    try:
        gpu_mem_old = get_gpu_profile()
        
        gc.collect()
        if str(DEVICE).__contains__("cuda"):
            torch.cuda.empty_cache()
        
        gpu_mem_new = get_gpu_profile()
        response = API_RESPONDER.result(is_successful=True, data={'GPU usage before': gpu_mem_old,
                                                                  'GPU usage after': gpu_mem_new, })
    except Exception as e:
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())
    
    return response


@app.post("/profile")
async def profile():
    try:
        gpu_mem = get_gpu_profile()
        sys_profile = get_cpu_info()
        sys_profile.update({ 'GPU usage': gpu_mem, })

        response = API_RESPONDER.result(is_successful=True, data=sys_profile)
    except Exception as e:
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())
    
    return response


@app.post("/upload")
async def upload(
        files: List[UploadFile] = File(description="Files to upload", 
                                        media_type='multipart/form-data'),
    ):
    """
    Test:
        multiple_files = [
            ('files', ('foo.fbx', open('foo.fbx', 'rb'))),
            ('files', ('bar.bvh', open('bar.bvh', 'rb'))),
        ]
        r = requests.post(url, files=multiple_files)
    """
    buffer = BytesIO()
    with ZipFile(buffer, mode='w', compression=ZIP_DEFLATED) as temp:
        for file in images:
            fcontent = await file.read()
            fname = ZipInfo(file.filename)
            temp.writestr(fname, fcontent)

    return StreamingResponse(
        iter([buffer.getvalue()]), 
        media_type="application/x-zip-compressed", 
        headers={ "Content-Disposition": "attachment; filename=files.zip"}
    )


if __name__ == "__main__":

    # Run application
    uvicorn.run(app, **API_CONFIG['SERVER'])

