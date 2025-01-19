import os
import io
import zipfile
import urllib.parse
import cv2
import numpy as np
from typing import List
from fastapi import FastAPI, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pathlib import Path
import json

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

def iterfile(file_path: str):
    with open(file_path, "rb") as f:
        yield from f

def tiff_to_png_stream(tiff_path: str):
    with Image.open(tiff_path) as img:
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        yield from img_buffer

@app.get("/images")
def get_image(filepath: str):
    decoded_path = urllib.parse.unquote(filepath)
    if not os.path.isfile(decoded_path):
        return {"error": "File not found"}
    ext = os.path.splitext(decoded_path)[1].lower()
    if ext in ('.tif', '.tiff'):
        return StreamingResponse(tiff_to_png_stream(decoded_path), media_type="image/png")
    else:
        return StreamingResponse(iterfile(decoded_path), media_type="image/png")

@app.get("/list_extensions")
def list_extensions(folder_path: str):
    decoded_path = urllib.parse.unquote(folder_path)
    exts = set()
    if not os.path.isdir(decoded_path):
        return []
    for root, dirs, files in os.walk(decoded_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext:
                exts.add(ext.lower())
    return sorted(list(exts))

def parse_filename(fn: str):
    """
    파일명: x좌표.y좌표.t.jpeg
    예) 12345.67890.t.jpeg
    => x = 12345, y= 67890
    없으면 0,0
    """
    base = os.path.basename(fn)
    # 확장자 전후로 split
    # e.g. "12345.67890.t.jpeg" -> "12345.67890.t"
    main_part, _ = os.path.splitext(base)
    # 혹시 .jpeg -> .jpg.split() 시 두 번 ext가 있으면 추가 파싱
    # 여기서는 단순히 rsplit('.', 2) 정도
    # "12345.67890.t" -> ["12345", "67890", "t"]
    parts = main_part.split('.')
    if len(parts) >= 3:
        try:
            x = float(parts[0])
            y = float(parts[1])
            # t = parts[2]
            return x, y
        except:
            return 0.0, 0.0
    return 0.0, 0.0

@app.get("/analyze_images")
def analyze_images(
    folder_path: str,
    extensions: List[str] = Query([]),
    algorithm: str = "laplacian_std"
):
    """
    - Laplacian STD(기본), Sobel Mean 등 선명도
    - GLV 평균, 표준편차 추가
    - 파일명 파싱 -> x,y (um)
    """
    decoded_path = urllib.parse.unquote(folder_path)
    results = []

    for root, dirs, files in os.walk(decoded_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in [e.lower() for e in extensions]:
                full_path = os.path.join(root, file)
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # 선명도 계산
                if algorithm == "laplacian_std":
                    lap = cv2.Laplacian(img, cv2.CV_64F)
                    score = lap.std()
                elif algorithm == "laplacian_var":
                    lap = cv2.Laplacian(img, cv2.CV_64F)
                    score = lap.var()
                elif algorithm == "sobel_mean":
                    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                    sobel = np.sqrt(sobelx**2 + sobely**2)
                    score = sobel.mean()
                elif algorithm == "10_90_rise":
                    sorted_pixels = np.sort(img.ravel())
                    p10 = np.percentile(sorted_pixels, 10)
                    p90 = np.percentile(sorted_pixels, 90)
                    score = float(p90 - p10)
                else:
                    score = 0.0

                # GLV mean, std
                glv_mean = float(np.mean(img))
                glv_std = float(np.std(img))

                # 파일명 -> x, y 좌표 (um)
                x_val, y_val = parse_filename(file)

                results.append({
                    "filename": file,
                    "extension": ext.lower(),
                    "fullPath": full_path.replace("\\", "/"),
                    "score": round(score, 4),
                    "glvMean": round(glv_mean, 2),
                    "glvStd": round(glv_std, 2),
                    "xCoord": x_val,  # um
                    "yCoord": y_val   # um
                })

    return results

@app.get("/download_project")
def download_project():
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("."):
            if "venv" in dirs:
                dirs.remove("venv")
            if ".git" in dirs:
                dirs.remove(".git")
            if ".idea" in dirs:
                dirs.remove(".idea")

            for file in files:
                if file.endswith(".zip"):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, ".")
                zf.write(file_path, arcname)

    mem_zip.seek(0)
    return StreamingResponse(mem_zip, media_type="application/x-zip-compressed",
                             headers={
                                 "Content-Disposition": "attachment; filename=FocusEval_Project.zip"
                             })

@app.get("/browse_folder")
async def browse_folder():
    try:
        # PowerShell 스크립트를 사용하여 폴더 선택 다이얼로그 실행
        ps_script = '''
        Add-Type -AssemblyName System.Windows.Forms
        $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
        $folderBrowser.Description = "폴더를 선택하세요"
        $result = $folderBrowser.ShowDialog()
        if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
            $folderBrowser.SelectedPath
        }
        '''
        
        # PowerShell 실행
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
            text=True
        )
        
        selected_path = result.stdout.strip()
        
        if selected_path:
            # 백슬래시를 포워드 슬래시로 변환
            selected_path = selected_path.replace('\\', '/')
            return {"status": "success", "path": selected_path}
            
        return {"status": "cancelled"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/get_extensions")
async def get_extensions(folder_path: str):
    try:
        extensions = set()
        for file in Path(folder_path).rglob('*'):
            if file.is_file():
                ext = file.suffix.lower()
                if ext:
                    extensions.add(ext[1:])
        return {"extensions": list(extensions)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
async def analyze_images(folder_path: str = Form(...), extensions: str = Form(...), algorithm: str = Form(...)):
    try:
        print(f"\n[분석 시작] 폴더: {folder_path}")
        print(f"선택된 확장자: {extensions}")
        print(f"선택된 알고리즘: {algorithm}")
        
        extensions_list = json.loads(extensions)
        decoded_path = urllib.parse.unquote(folder_path)
        results = []
        
        # 전체 파일 수 계산
        total_files = sum(1 for root, _, files in os.walk(decoded_path)
                         for file in files
                         if os.path.splitext(file)[1].lower()[1:] in extensions_list)
        
        processed_files = 0
        print(f"\n총 {total_files}개 파일 처리 예정")

        for root, _, files in os.walk(decoded_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower()[1:] in extensions_list:
                    processed_files += 1
                    print(f"\r진행중: {processed_files}/{total_files} ({int(processed_files/total_files*100)}%)", end="")
                    
                    full_path = os.path.join(root, file)
                    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"\n경고: {file} 파일을 읽을 수 없습니다.")
                        continue

                    # 선명도 계산
                    if algorithm == "laplacian_std":
                        lap = cv2.Laplacian(img, cv2.CV_64F)
                        score = lap.std()
                    elif algorithm == "laplacian_var":
                        lap = cv2.Laplacian(img, cv2.CV_64F)
                        score = lap.var()
                    elif algorithm == "sobel_mean":
                        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                        sobel = np.sqrt(sobelx**2 + sobely**2)
                        score = sobel.mean()
                    elif algorithm == "10_90_rise":
                        sorted_pixels = np.sort(img.ravel())
                        p10 = np.percentile(sorted_pixels, 10)
                        p90 = np.percentile(sorted_pixels, 90)
                        score = float(p90 - p10)
                    else:
                        score = 0.0

                    # GLV mean, std
                    glv_mean = float(np.mean(img))
                    glv_std = float(np.std(img))

                    # 파일명 -> x, y 좌표 (um)
                    x_val, y_val = parse_filename(file)

                    results.append({
                        "filename": file,
                        "path": full_path,
                        "score": score,
                        "glv_mean": glv_mean,
                        "glv_std": glv_std,
                        "x": x_val,
                        "y": y_val
                    })

        print(f"\n\n분석 완료! {len(results)}개 파일 처리됨")
        return {"results": results}
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        return {"error": str(e)}
