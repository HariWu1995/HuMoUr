import sys
sys.path.append('./')

import os
import gc
import random
import argparse
import traceback
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
from typing import Union, Literal, List, Tuple, Optional

import PIL
from PIL import Image
from io import BytesIO

import torch
import numpy as np

import uvicorn

from fastapi import FastAPI, Form, File, UploadFile, Request, status
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse, Response
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware

from api.templates import OutputAPI
from api.action_names import all_actions

from src.utils.io_util import load_config, load_multipart_file
from src.utils.dtype_util import image_to_base64
from src.utils.profiler_util import get_gpu_memory, get_gpu_profile, get_cpu_info


#############################
#       Configuration       #
#############################

DATASET_0_DIR = '/kaggle/input/humanml3d/HumanML3D/humanml'
DATASET_1_DIR = '/kaggle/input/humodimo-datasets/dataset'
DATASET_2_DIR = '/kaggle/input/humodimo-datasets-2/dataset'

CHECKPOINT_DIR = '/kaggle/input/humodimo-checkpoints/checkpoints'
DEPENDENCY_DIR = "/kaggle/input/humodimo"
SMPL_DATA_ROOT = "/kaggle/input/smpl-human-models/body_models"

if os.environ.get('DEPENDENCY_DIR', None) is None:
    os.environ["DEPENDENCY_DIR"] = DEPENDENCY_DIR

if os.environ.get('SMPL_DATA_ROOT', None) is None:
    os.environ["SMPL_DATA_ROOT"] = SMPL_DATA_ROOT

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
    humodel_t2m_opt = { 'humanml_trans_dec_512': 375_000,
                        'humanml_trans_enc_512': 475_000, },
    humodel_t2mm_opt = {'humanml_trans_enc_512': 200_000,  
                       'Babel_TrasnEmb_GeoLoss': 850_000, },
    humodel_t2cm_opt = {'pw3d_prefix' : 240_000, 
                        'pw3d_text'   : 100_000, },
    humodel_ctrl_opt = {  'left_wrist': 280_000,
                          'right_foot': 280_000,
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
    animodel_m2m_opt = {  'horse':  39_999, 'jaguar': 130_000,
                        'ostrich': 140_000, 'wyvern': None, },
)

DATASET_CONFIG = dict(
    HumanML_full        = f"{DATASET_0_DIR}",
    HumanML_small       = f"{DATASET_1_DIR}/HumanML3D",
    HumanAct12Poses     = f"{DATASET_1_DIR}/HumanAct12Poses",
    KIT_ML              = f"{DATASET_1_DIR}/KIT-ML",
    UESTC_RGBD          = f"{DATASET_1_DIR}/uestc",
    BABEL               = f"{DATASET_2_DIR}/babel",
    PosesWild3D         = f"{DATASET_2_DIR}/3dpw",
    TruebonesZoo        = f"{DATASET_2_DIR}/zoo",
    Mixamo              = f"{DATASET_2_DIR}/mixamo",
)

ANIMODEL_MAP = dict(
      horse = "HorseALL-LoneRanger",
     jaguar = "Jaguar-Attack",
    ostrich = "Ostrich-Attack3",
     wyvern = "Wyvern-Fly",
)


#############################
#         API Setup         #
#############################

API_RESPONDER = OutputAPI()
API_CONFIG = load_config(path='./api/config.yaml')

app = FastAPI(
      root_path =  os.getenv("ROOT_PATH"), 
          title = API_CONFIG['INFO']['title'],
    description = API_CONFIG['INFO']['intro'],
   openapi_tags = API_CONFIG['TAGS'],
        version = API_CONFIG['VERSION'],
       docs_url = None, 
      redoc_url = None,
)

app.add_middleware(
    CORSMiddleware,         # https://fastapi.tiangolo.com/tutorial/cors/
    allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
    allow_origins=['*'],    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
    allow_methods=['*'],    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
    allow_headers=['*'],    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")


#############################
#       API Endpoints       #
#############################

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):

    # root_path = app.root_path
    root_path = request.scope.get("root_path", "").rstrip("/")
    query_params = str(request.query_params)

    openapi_url = app.openapi_url
    if openapi_url:
        openapi_url = root_path + openapi_url + "?" + query_params

    oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url
    if oauth2_redirect_url:
        oauth2_redirect_url = root_path + oauth2_redirect_url

    return get_swagger_ui_html(
                      title   = app.title + " - Swagger UI",
                openapi_url   = openapi_url,
        oauth2_redirect_url   = oauth2_redirect_url,
                 init_oauth   = app.swagger_ui_init_oauth,
        swagger_ui_parameters = app.swagger_ui_parameters,
        swagger_favicon_url   = "/assets/favicon.ico",
    )


@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.post("/config/dependencies", tags=["Configuration"])
async def _config_dependencies(
        DEPENDENCY_DIR : str = Form(description=API_CONFIG['ARGUMENTS']['DEPENDENCY_DIR'], default=DEPENDENCY_DIR), 
        SMPL_DATA_ROOT : str = Form(description=API_CONFIG['ARGUMENTS']['SMPL_DATA_ROOT'], default=SMPL_DATA_ROOT), 
    ):
    try:
        os.environ["SMPL_DATA_ROOT"] = SMPL_DATA_ROOT
        os.environ["DEPENDENCY_DIR"] = DEPENDENCY_DIR

        new_config = dict(  
                        DEPENDENCY_DIR = DEPENDENCY_DIR,
                        SMPL_DATA_ROOT = SMPL_DATA_ROOT,
                    )

        response = API_RESPONDER.result(is_successful=True, data=new_config)

    except Exception as e:
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())
    
    return response


@app.post("/config/models", tags=["Configuration"])
async def _config_models(
          MODEL_DIR :  str = Form(description=API_CONFIG['ARGUMENTS'][ 'MODEL_DIR' ], default=MODEL_CONFIG['MODEL_DIR']), 
        cuda_device :  int = Form(description=API_CONFIG['ARGUMENTS']['cuda_device'], default=MODEL_CONFIG['cuda_device']), 
        cpu_offload : bool = Form(description=API_CONFIG['ARGUMENTS']['cpu_offload'], default=MODEL_CONFIG['cpu_offload']), 
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


@app.post("/config/datasets", tags=["Configuration"])
async def _config_datasets(
        HumanML_full    : str = Form(default=DATASET_CONFIG['HumanML_full']), 
        HumanML_small   : str = Form(default=DATASET_CONFIG['HumanML_small']), 
        HumanAct12Poses : str = Form(default=DATASET_CONFIG['HumanAct12Poses']), 
        KIT_ML          : str = Form(default=DATASET_CONFIG['KIT_ML']), 
        UESTC_RGBD      : str = Form(default=DATASET_CONFIG['UESTC_RGBD']), 
        BABEL           : str = Form(default=DATASET_CONFIG['BABEL']), 
        PosesWild3D     : str = Form(default=DATASET_CONFIG['PosesWild3D']), 
        TruebonesZoo    : str = Form(default=DATASET_CONFIG['TruebonesZoo']), 
        Mixamo          : str = Form(default=DATASET_CONFIG['Mixamo']), 
    ):
    try:
        global DATASET_CONFIG
        DATASET_CONFIG.update(dict(
            HumanML_full = HumanML_full,
            HumanML_small = HumanML_small,
            HumanAct12Poses = HumanAct12Poses,
                PosesWild3D = PosesWild3D,
               TruebonesZoo = TruebonesZoo,
                    KIT_ML = KIT_ML,
                UESTC_RGBD = UESTC_RGBD,
                     BABEL = BABEL,
                    Mixamo = Mixamo,
        ))
        response = API_RESPONDER.result(is_successful=True, data=DATASET_CONFIG)

    except Exception as e:
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())
    
    return response


@app.post("/api/generate", tags=["Main"])
async def _api_generate(
        action: Literal[all_actions] = \
                        Form(description=API_CONFIG['ARGUMENTS']['action_name'], default=None), 
        prompt:  str = Form(description=API_CONFIG['ARGUMENTS']['prompt_positive'], default=None), 
      is_multi: bool = Form(description=API_CONFIG['ARGUMENTS']['prompt_multiple'], default=False), 
    model_ckpt: Literal[] = \
                        Form(description=API_CONFIG['ARGUMENTS']['model_ckpt'], default=False), 
    ):

    try:
        print(f"\n\n\naction = {action} - prompt = {prompt}")

        # Load arguments and pipeline
        if is_multi:
            from src.mdm_prior.sequentialize import main as generate_pipe
            from src.mdm_prior.utils.parser_util import generate_args

        else:
            from src.mdm.generate import main as generate_pipe
            from src.mdm.utils.parser_util import generate_args

        # Run pipeline
        if prompt is not None:

            if is_multi:
                print('\nRunning text-to-motions ...')

            else:
                print('\nRunning text-to-motion ...')

        elif action is not None:
            print('\nRunning action-to-motion ...')

        else:
            raise ValueError('Either `action` or `prompt` MUST be NOT null')

        response = API_RESPONDER.result(is_successful=True, data={})
        
    except Exception as e:
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())

    return response


@app.post("/resources/clear", tags=["Resources"])
async def _resources_clear():
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


@app.post("/resources/profile", tags=["Resources"])
async def _resources_profile():
    try:
        gpu_mem = get_gpu_profile()
        sys_profile = get_cpu_info()
        sys_profile.update({ 'GPU usage': gpu_mem, })

        response = API_RESPONDER.result(is_successful=True, data=sys_profile)
    except Exception as e:
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())
    
    return response


@app.post("/test/upload", tags=["Test"])
async def _test_upload(
        files: List[UploadFile] = File(description="Files to upload", media_type='multipart/form-data'),
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
        headers={"Content-Disposition": "attachment; filename=files.zip"}
    )


if __name__ == "__main__":

    # Run application
    uvicorn.run(app, **API_CONFIG['SERVER'])

