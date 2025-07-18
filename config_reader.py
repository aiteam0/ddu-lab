import json
import os

from loguru import logger

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = None

CONFIG_FILE_NAME_ENV = os.getenv('MINERU_TOOLS_CONFIG_JSON', 'magic-pdf.json')

def find_config_file():
    if os.path.isabs(CONFIG_FILE_NAME_ENV):
        if os.path.exists(CONFIG_FILE_NAME_ENV):
            logger.debug(f"절대 경로에서 설정 파일 찾음: {CONFIG_FILE_NAME_ENV}")
            return CONFIG_FILE_NAME_ENV
        else:
            logger.warning(f"환경 변수에 지정된 절대 경로에 파일 없음: {CONFIG_FILE_NAME_ENV}. 다른 위치를 탐색합니다.")
            config_basename = os.path.basename(CONFIG_FILE_NAME_ENV)
    else:
        config_basename = CONFIG_FILE_NAME_ENV

    cwd_path = os.path.join(os.getcwd(), config_basename)
    if os.path.exists(cwd_path):
        logger.debug(f"현재 작업 디렉토리에서 설정 파일 찾음: {cwd_path}")
        return cwd_path

    if SCRIPT_DIR:
        script_dir_path = os.path.join(SCRIPT_DIR, config_basename)
        if os.path.exists(script_dir_path):
            logger.debug(f"스크립트 디렉토리에서 설정 파일 찾음: {script_dir_path}")
            return script_dir_path

    return None

def read_config():
    config_file_path = find_config_file()

    if not config_file_path:
        searched_paths = []
        config_basename = os.path.basename(CONFIG_FILE_NAME_ENV) # 실제 찾으려던 파일 이름
        if os.path.isabs(CONFIG_FILE_NAME_ENV): searched_paths.append(CONFIG_FILE_NAME_ENV)
        searched_paths.append(os.path.join(os.getcwd(), config_basename))
        if SCRIPT_DIR: searched_paths.append(os.path.join(SCRIPT_DIR, config_basename))

        raise FileNotFoundError(
            f"설정 파일 '{config_basename}'을 찾을 수 없습니다. "
            f"검색한 위치: {', '.join(list(dict.fromkeys(searched_paths)))}" # 중복 제거
        )

    logger.info(f"설정 파일 읽는 중: {config_file_path}")
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"{config_file_path} 파일의 JSON 형식이 잘못되었습니다: {e}")
    except Exception as e:
        raise IOError(f"{config_file_path} 파일 읽기 중 오류 발생: {e}")


def parse_bucket_key(path: str):
    if path.startswith("s3://"):
        parts = path[5:].split('/', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            return parts[0], ""
    return None, None

class MODEL_NAME:
    LAYOUTLMv3 = "LayoutLMv3"
    YOLO_V8_MFD = "YOLOv8_MFD"
    UniMerNet_v2_Small = "UniMerNet_v2_Small"
    RAPID_TABLE = "RapidTable"

def get_s3_config(bucket_name: str):
    config = read_config()
    bucket_info = config.get('bucket_info', {})

    default_config = bucket_info.get('[default]')

    bucket_specific_config = bucket_info.get(bucket_name)

    if bucket_specific_config:
        access_key, secret_key, storage_endpoint = bucket_specific_config
    elif default_config:
        logger.debug(f"버킷 '{bucket_name}'에 대한 특정 설정 없음. 기본 설정을 사용합니다.")
        access_key, secret_key, storage_endpoint = default_config
    else:
        raise KeyError(f"버킷 '{bucket_name}' 또는 '[default]'에 대한 S3 설정이 {CONFIG_FILE_NAME_ENV} 파일에 없습니다.")

    if access_key is None or secret_key is None or storage_endpoint is None:
        raise ValueError(f"버킷 '{bucket_name}' 또는 '[default]' 설정에 null 값이 포함되어 있습니다 ({CONFIG_FILE_NAME_ENV} 파일).")

    return access_key, secret_key, storage_endpoint


def get_s3_config_dict(path: str):
    bucket_name = get_bucket_name(path)
    if not bucket_name:
         raise ValueError(f"경로에서 버킷 이름을 추출할 수 없습니다: {path}")
    access_key, secret_key, storage_endpoint = get_s3_config(bucket_name)
    return {'ak': access_key, 'sk': secret_key, 'endpoint': storage_endpoint}


def get_bucket_name(path):
    bucket, key = parse_bucket_key(path)
    return bucket


def get_local_models_dir():
    config = read_config()
    models_dir = config.get('models-dir')
    if models_dir is None:
        default_path = '/tmp/models' # 기본 경로 설정
        logger.warning(f"'models-dir' 설정을 찾을 수 없습니다. 기본값 '{default_path}'를 사용합니다.")
        return default_path
    return models_dir


def get_local_layoutreader_model_dir():
    config = read_config()
    layoutreader_model_dir = config.get('layoutreader-model-dir')
    if layoutreader_model_dir:
        return layoutreader_model_dir
    else:
        models_dir = get_local_models_dir()
        default_path = os.path.join(models_dir, 'layoutreader')

        logger.warning(f"'layoutreader-model-dir' 설정을 찾을 수 없습니다. 기본 경로 '{default_path}'를 가정합니다.")
        return default_path


def get_device():
    config = read_config()
    device = config.get('device-mode')
    if device is None:
        default_device = 'cpu'
        logger.warning(f"'device-mode' 설정을 찾을 수 없습니다. 기본값 '{default_device}'를 사용합니다.")
        return default_device
    else:
        return device


def _get_config_with_default(config_key: str, default_json_str: str, model_name_for_default: str = ""):
    config = read_config()
    specific_config = config.get(config_key)
    if specific_config is not None:
        return specific_config
    else:
        try:
            formatted_default_str = default_json_str.format(model_name=model_name_for_default)
            default_config = json.loads(formatted_default_str)
            logger.warning(f"'{config_key}' 설정을 찾을 수 없습니다. 기본값을 사용합니다: {default_config}")
            return default_config
        except (KeyError, json.JSONDecodeError) as e:
             logger.error(f"'{config_key}'의 기본 설정 로드 중 오류 발생: {e}. 빈 딕셔너리를 반환합니다.")
             return {}


def get_table_recog_config():
    return _get_config_with_default(
        'table-config',
        '{{"model": "{model_name}","enable": false, "max_time": 400}}',
        MODEL_NAME.RAPID_TABLE
    )


def get_layout_config():
     return _get_config_with_default(
        'layout-config',
        '{{"model": "{model_name}"}}',
        MODEL_NAME.LAYOUTLMv3
     )


def get_formula_config():
     default_str = f'{{"mfd_model": "{MODEL_NAME.YOLO_V8_MFD}","mfr_model": "{MODEL_NAME.UniMerNet_v2_Small}","enable": true}}'
     return _get_config_with_default('formula-config', default_str)


def get_llm_aided_config():
    config = read_config()
    llm_aided_config = config.get('llm-aided-config')
    if llm_aided_config is None:
        logger.warning(f"'llm-aided-config' 설정을 찾을 수 없습니다. 기본값 'None'을 사용합니다.")
        return None
    else:
        return llm_aided_config


if __name__ == '__main__':
    try:
        print("--- 설정 파일 읽기 테스트 ---")
        cfg = read_config()
        print(f"성공적으로 설정을 읽었습니다: {cfg}")

        print("\n--- 로컬 모델 디렉토리 테스트 ---")
        models_dir = get_local_models_dir()
        print(f"모델 디렉토리: {models_dir}")
        layout_dir = get_local_layoutreader_model_dir()
        print(f"LayoutReader 디렉토리: {layout_dir}")

        print("\n--- S3 설정 테스트 (llm-raw 버킷) ---")
        try:
            ak, sk, endpoint = get_s3_config('llm-raw')
            print(f"S3 설정 (llm-raw): AK={ak[:4]}..., SK=***, Endpoint={endpoint}")
        except (KeyError, ValueError) as e:
            print(f"S3 설정 가져오기 실패 (llm-raw): {e}")

        print("\n--- 장치 설정 테스트 ---")
        device = get_device()
        print(f"장치 설정: {device}")

        print("\n--- 테이블 인식 설정 테스트 ---")
        table_cfg = get_table_recog_config()
        print(f"테이블 설정: {table_cfg}")

    except FileNotFoundError as e:
        print(f"설정 파일 찾기 실패: {e}")
    except Exception as e:
        print(f"테스트 중 예상치 못한 오류 발생: {e}")