import os
import openai
import dotenv
import datetime

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

dotenv.load_dotenv()

# 创建 FastAPI 应用程序
app = FastAPI()

# 设置您的 API 密钥
api_key = os.getenv("API_KEY")

# 定义输出目录
output_dir = Path("outputs")
# 创建输出目录，如果不存在则创建
output_dir.mkdir(parents=True, exist_ok=True)

# 定义请求模型类
class GenerateImagesRequest(BaseModel):
    model: str = "dall-e-3"
    size: str = "1024x1024"
    style: str = "vivid"
    quality: str = "standard"
    n: int = 1
    prompt: str


# 定义生成并保存图像的函数
async def generate_and_save_image(
    index: int, generate_image_request: GenerateImagesRequest
):
    # 创建 OpenAI 客户端对象
    client = openai.OpenAI(api_key=api_key, base_url=os.getenv("base_url"))
    try:
        # 调用 OpenAI API 生成图像
        response_image = client.images.generate(
            prompt=generate_image_request.prompt,
            model=generate_image_request.model,
            n=1,
            size=generate_image_request.size,
            style=generate_image_request.style,
            quality=generate_image_request.quality,
        )
        # 获取图像的 URL 和修正后的 prompt
        image_url = response_image.data[0].url
        revised_prompt = getattr(
            response_image.data[0], "revised_prompt", "no-revised-prompt"
        )
        # 下载图像内容
        # image_bytes = requests.get(image_url).content
        # 返回图像并返回相关数据
        return (
            {
                "url": image_url,
                "revised_prompt": revised_prompt,
                # "image_bytes": image_bytes,
            },
            200,
            None,
        )
    except openai.APIError as e:
        # 如果发生错误，返回状态码和错误信息
        return None, e.http_status, str(e)


@app.post("/images/generations")
async def dalle3_drawing(request: GenerateImagesRequest):
    num_images = 1

    # 存储生成失败的任务
    errors = []
    images_data = []
    # 逐个生成图像
    for index in range(num_images):
        # 获取任务执行结果（状态码和错误信息）
        image_data, status_code, error_message = await generate_and_save_image(
            index, request
        )
        # 如果状态码不为 200，输出错误信息
        if status_code != 200:
            errors.append(f"Failed to generate image {index + 1}: {error_message}")
        else:
            # 如果成功，将图像数据添加到列表中
            images_data.append(image_data)

    if errors:
        raise HTTPException(status_code=500, detail=errors)

    # 根据 OpenAI 的返回格式，返回图像的 URL 和其他相关信息
    response = {
        "created": datetime.datetime.now().isoformat(),
        "data": [
            {"url": image_data["url"], "revised_prompt": image_data["revised_prompt"]}
            for image_data in images_data
        ],
    }

    return response

@app.post("/apikey")
async def set_api_key(new_api_key: str):
    global api_key
    api_key = new_api_key
    return {"message": f"API key set successfully, new key: {api_key}"}
# 启动 FastAPI 应用程序时，可以通过 uvicorn 运行
# 例如：
# uvicorn image:app --reload
