"""沙箱文件 API：列出 / 下载沙箱工作目录中的文件（沙箱产物持久化到宿主机后供用户下载）。"""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])


def _work_dir() -> Path:
    """沙箱/宿主机共享工作目录（绝对路径，自动创建）。与 tools/run_command._work_dir 保持一致。"""
    d = Path(settings.sandbox_work_dir).expanduser()
    if not d.is_absolute():
        d = Path(__file__).resolve().parents[2] / d
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def _safe_path(rel: str) -> Path:
    """把相对路径安全解析到工作目录内，拒绝路径穿越（..、绝对路径等）。"""
    work = _work_dir()
    raw = (work / rel).resolve()
    if not raw.is_relative_to(work):
        raise HTTPException(status_code=400, detail="非法的文件路径")
    return raw


@router.get("/files")
async def list_files():
    """列出沙箱工作目录中的文件（相对路径 + 大小 + 修改时间），供前端渲染下载列表。"""
    work = _work_dir()
    items = []
    for p in sorted(work.rglob("*")):
        if p.is_file():
            st = p.stat()
            items.append(
                {
                    "path": str(p.relative_to(work)).replace("\\", "/"),
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                }
            )
    return {"files": items, "work_dir": str(work)}


@router.get("/files/download")
async def download(path: str = Query(...)):
    """下载沙箱工作目录中的指定文件（path 为相对路径，如 work/report.txt）。"""
    raw = _safe_path(path)
    if not raw.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(raw, filename=raw.name, media_type="application/octet-stream")
