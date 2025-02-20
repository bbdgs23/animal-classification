import os
import replicate
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import base64
from pydantic import BaseModel
from dotenv import load_dotenv

# 환경변수 로드 (있는 경우)
load_dotenv(dotenv_path='.env', verbose=True)

app = FastAPI()

# CORS 설정 - 프론트엔드 도메인 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (테스트용, 프로덕션에서는 특정 도메인만 허용하세요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replicate API 키 설정
replicate_token = os.getenv("REPLICATE_API_TOKEN", "your_replicate_api_token")
os.environ["REPLICATE_API_TOKEN"] = replicate_token

class AnimalClassificationResponse(BaseModel):
    classified: dict = {}
    unclassified: list = []

# 동물 종류 분류 함수
async def classify_animal(image_data):
    """동물 종류를 분류하는 함수"""
    try:
        # Replicate의 CLIP 기반 분류 모델 사용
        output = replicate.run(
            "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
            input={
                "image": image_data,
                "question": "What type of animal is this? Options: dog, cat, rabbit, bear, iguana, bird, fish, other"
            }
        )
        result = output.strip().lower()
        
        # 모델이 분류를 못했거나 응답이 없는 경우 '미분류'로 처리
        if not result or result == "other" or "cannot" in result or "not sure" in result or "unclear" in result:
            return "unclassified"
        return result
    except Exception as e:
        # 예외 발생 시 '미분류'로 처리
        print(f"분류 모델 오류: {str(e)}")
        return "unclassified"

# 강아지/고양이 품종 분류 함수
async def classify_breed(image_data, animal_type):
    """강아지나 고양이 품종을 분류하는 함수"""
    if animal_type not in ["dog", "cat"]:
        return None
    
    try:
        # 강아지 품종 분류 모델
        if animal_type == "dog":
            output = replicate.run(
                "modelsaggregation/image-classification:21ceaa9c218d3a6fd09c1c6831fc1c4a5ca3ac820cd09bdf2110b3aacf47e3ba",
                input={
                    "image": image_data,
                    "model_name": "dog-breed-identification",
                }
            )
        # 고양이 품종 분류 모델
        else:
            output = replicate.run(
                "modelsaggregation/image-classification:21ceaa9c218d3a6fd09c1c6831fc1c4a5ca3ac820cd09bdf2110b3aacf47e3ba",
                input={
                    "image": image_data, 
                    "model_name": "cat-breed-classification",
                }
            )
        
        # 결과 파싱
        if output and len(output) > 0:
            # 첫 번째 결과 (가장 높은 확률)
            breed = output[0]['label']
            confidence = output[0]['confidence']
            return {"breed": breed, "confidence": confidence}
            
        return None
    except Exception as e:
        print(f"품종 분류 오류: {str(e)}")
        return None

@app.post("/predict", response_model=AnimalClassificationResponse)
async def predict(file: UploadFile = File(...)):
    """단일 이미지를 분류하고 프론트엔드 포맷에 맞게 응답하는 엔드포인트"""
    try:
        # 이미지 데이터 읽기
        image_content = await file.read()
        
        # 이미지 데이터를 base64로 인코딩
        encoded_image = base64.b64encode(image_content).decode("utf-8")
        base64_image = f"data:image/{file.content_type.split('/')[-1]};base64,{encoded_image}"
        
        # 동물 종류 분류
        animal_type = await classify_animal(base64_image)
        
        result = AnimalClassificationResponse()
        
        # 품종 분류 (강아지/고양이인 경우)
        if animal_type in ["dog", "cat"]:
            breed_info = await classify_breed(base64_image, animal_type)
            
            if breed_info and breed_info["breed"]:
                # 분류 결과 저장
                breed_name = breed_info["breed"]
                confidence = breed_info["confidence"]
                
                # 프론트엔드 형식에 맞게 결과 구성
                result.classified[breed_name] = [{
                    "confidence": confidence,
                    "breed": breed_name,
                    "animal_type": animal_type
                }]
            else:
                # 품종 분류 실패
                result.unclassified = [{}]
        else:
            # 강아지/고양이가 아님
            result.unclassified = [{}]
            
        return result
            
    except Exception as e:
        # 예외 발생 시 빈 결과 반환
        print(f"이미지 처리 오류: {str(e)}")
        return AnimalClassificationResponse(unclassified=[{}])

# 호환성을 위한 다중 이미지 처리 엔드포인트
@app.post("/batch-predict", response_model=AnimalClassificationResponse)
async def batch_predict(files: List[UploadFile] = File(...)):
    """여러 이미지를 한번에 분류하는 엔드포인트"""
    result = AnimalClassificationResponse()
    classified_data = {}
    unclassified_images = []
    
    for file in files:
        try:
            # 이미지 데이터 읽기
            image_content = await file.read()
            
            # 이미지 데이터를 base64로 인코딩
            encoded_image = base64.b64encode(image_content).decode("utf-8")
            base64_image = f"data:image/{file.content_type.split('/')[-1]};base64,{encoded_image}"
            
            # 동물 종류 분류
            animal_type = await classify_animal(base64_image)
            
            # 품종 분류 (강아지/고양이인 경우)
            if animal_type in ["dog", "cat"]:
                breed_info = await classify_breed(base64_image, animal_type)
                
                if breed_info and breed_info["breed"]:
                    # 분류 결과 저장
                    breed_name = breed_info["breed"]
                    confidence = breed_info["confidence"]
                    
                    # 이미 있는 품종이면 리스트에 추가
                    if breed_name in classified_data:
                        classified_data[breed_name].append({
                            "confidence": confidence,
                            "breed": breed_name,
                            "animal_type": animal_type
                        })
                    else:
                        # 새 품종이면 새로운 리스트 생성
                        classified_data[breed_name] = [{
                            "confidence": confidence,
                            "breed": breed_name,
                            "animal_type": animal_type
                        }]
                else:
                    # 품종 분류 실패
                    unclassified_images.append({})
            else:
                # 강아지/고양이가 아님
                unclassified_images.append({})
                
        except Exception as e:
            print(f"이미지 처리 오류: {str(e)}")
            unclassified_images.append({})
    
    result.classified = classified_data
    result.unclassified = unclassified_images
    return result

@app.get("/")
async def root():
    """루트 경로 핸들러 - 서버 상태 확인용"""
    return {"status": "online", "message": "동물 분류 API가 실행 중입니다"}

if __name__ == "__main__":
    import uvicorn
    # 포트를 5050으로 고정
    port = 5050
    uvicorn.run(app, host="0.0.0.0", port=port)
