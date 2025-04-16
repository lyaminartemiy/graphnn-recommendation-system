import boto3
from fastapi.responses import JSONResponse
import asyncio
from fastapi import APIRouter, Depends
from src.lifespan import get_s3_client
import base64
from fastapi import HTTPException
from fastapi.responses import Response
from fastapi import Query


router = APIRouter(prefix="/recommendations")


@router.get("/product_images/")
async def get_images_bytes(
    article_ids: list[str] = Query(...),
    s3_client: boto3.client = Depends(get_s3_client),
) -> Response:
    async def get_image_bytes(article_id):
        try:
            s3_key = f"0{article_id}.jpg"
            response = s3_client.get_object(
                Bucket="product-images",
                Key=s3_key
            )
            image_bytes = response['Body'].read()
            return {article_id: image_bytes}
        except s3_client.exceptions.NoSuchKey:
            return {article_id: None}
        except Exception as e:
            print(f"Error fetching image {article_id}: {str(e)}")
            return {article_id: None}

    # Параллельная обработка ID
    tasks = [get_image_bytes(article_id) for article_id in article_ids]
    results = await asyncio.gather(*tasks)
    
    # Собираем все успешные результаты
    images_data = dict(pair for d in results for pair in d.items() if pair[1] is not None)
    
    if not images_data:
        raise HTTPException(status_code=404, detail="No images found")
    
    encoded_images = {
        article_id: base64.b64encode(img_bytes).decode('utf-8') 
        for article_id, img_bytes in images_data.items()
    }
    
    return JSONResponse(content=encoded_images)
