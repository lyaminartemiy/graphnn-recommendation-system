import uvicorn
from fastapi import FastAPI
from src.lifespan import lifespan
from src.routers.events import router as events_router
from src.routers.personal_items import router as personal_items_router
from src.routers.product_images import router as product_items_router

app = FastAPI(lifespan=lifespan)

app.include_router(events_router)
app.include_router(personal_items_router)
app.include_router(product_items_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
