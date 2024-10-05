import uvicorn
import glob
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import json
import os

app = FastAPI()
IMAGE_DIR = 'data/'

def get_latest_images(num_images=20):
    results = []
    image_files = glob.glob("./data/*.jpg")
    num_images = min(num_images, len(image_files))
    # json_files = glob.glob("./data/*.json")
    image_files.sort(reverse=True)
    for i in range(num_images):
        timestr = Path(image_files[i]).stem
        json_fp = f'./data/{timestr}.json'
        with open(json_fp, 'r') as f:
            objects = json.load(f)
        results.append((timestr + '.jpg', str(objects)))
    return results

        

@app.get("/", response_class=HTMLResponse)
async def root():
    files = get_latest_images()
    images_html = ''.join(
        f'''<img src="/images/{img_file}" alt="{img_file}" style="width:200px; margin:12px;">
          <p> {objects} </p>
        '''
        for img_file, objects in files
    )
    return f"""
    <html>
        <head>
            <title>Xiao mao cam</title>
        </head>
        <body>
            <h1>Xiao mao meow</h1>
            <div>{images_html}</div>
        </body>
    </html>
    """

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = os.path.join(IMAGE_DIR, image_name)
    if os.path.isfile(image_path):
        return FileResponse(image_path)
    else:
        return {'error': 'Image not found'}, 404

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)