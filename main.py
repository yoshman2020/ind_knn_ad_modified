from indad.data import IMAGENET_MEAN, IMAGENET_STD
from indad.models import SPADE, PaDiM, PatchCore, KNNExtractor
from indad.data import MVTecDataset, StreamingDataset
from scipy import stats
from typing import Dict, Any, Type
from PIL import Image
import io
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import uuid

from fastapi import FastAPI, Request, Form, UploadFile, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
import inspect
import base64
from pydantic.fields import ModelField

matplotlib.use("Agg")


def as_form(cls: Type[BaseModel]):
    new_parameters = []

    for field_name, model_field in cls.__fields__.items():
        model_field: ModelField  # type: ignore

        new_parameters.append(
            inspect.Parameter(
                model_field.alias,
                inspect.Parameter.POSITIONAL_ONLY,
                default=Form(...)
                if model_field.required
                else Form(model_field.default),
                annotation=model_field.outer_type_,
            )
        )

    async def as_form_func(**data):
        return cls(**data)

    sig = inspect.signature(as_form_func)
    sig = sig.replace(parameters=new_parameters)
    as_form_func.__signature__ = sig  # type: ignore
    setattr(cls, "as_form", as_form_func)
    return cls


@as_form
class PredictForm(BaseModel):
    """検査フォーム

    Args:
        BaseModel (BaseModel): BaseModel
    """

    # 正常画像
    file_train: list[UploadFile] = None
    # 検査画像
    file_test: list[UploadFile] = None
    # サンプルデータ使用
    chk_sample: bool | None = None
    # MVTecデータセット
    sel_mvtec: str | None = None
    # method
    sel_method: str | None = None
    # backbone
    sel_backbone: str | None = None
    # モデル読込み
    chk_model: bool | None = None
    # モデルファイル
    file_model: UploadFile | None = None
    # 色範囲最小値
    hdn_color_min: int | None = None
    # 色範囲最大値
    hdn_color_max: int | None = None
    # 色範囲に相対値使用
    chk_relative: bool | None = None
    # しきい値
    rng_threshold: int | None = None
    # しきい値％
    num_threshold_rate: float | None = None
    # スコア最小値
    rng_threshold_min: int | None = None
    # スコア最大値
    rng_threshold_max: int | None = None
    # 入力値変更有
    hdn_value_changed: bool | None = None


class SessionData:
    """セッションで使用するデータ"""

    # セッションID
    session_id: str = ""
    # モデル
    model: KNNExtractor | None = None
    # 検査データセット
    test_dataset: StreamingDataset | None = None
    # 検査スコア
    pxl_lvl_anom_score: torch.Tensor | None = None
    # 現在データ行
    current_index: int = 0


# セッションで使用するモデル
session_data = SessionData()


def load_model(sel_method, sel_backbone):
    """モデル読込み

    Args:
        sel_method (str): method
        sel_backbone (str): backbone

    Returns:
        KNNExtractor: モデル
    """
    if sel_method == "SPADE":
        model = SPADE(
            k=3,
            backbone_name=sel_backbone,
        )
    elif sel_method == "PaDiM":
        model = PaDiM(
            d_reduced=75,
            backbone_name=sel_backbone,
        )
    elif sel_method == "PatchCore":
        model = PatchCore(
            f_coreset=0.01,
            backbone_name=sel_backbone,
            coreset_eps=0.95,
        )

    return model


def tensor_to_img(x: torch.Tensor, normalize=False) -> torch.Tensor:
    """Tensorを画像変換

    Args:
        x (torch.Tensor): Tensor
        normalize (bool, optional): 正規化有無. Defaults to False.

    Returns:
        torch.Tensor: 画像Tensor
    """
    if normalize:
        x *= IMAGENET_STD.unsqueeze(-1).unsqueeze(-1)
        x += IMAGENET_MEAN.unsqueeze(-1).unsqueeze(-1)
    x = x.clip(0.0, 1.0).permute(1, 2, 0).detach().numpy()
    return x


def pred_to_img(x: torch.Tensor, range: tuple[int, int]) -> torch.Tensor:
    """結果画像変換

    Args:
        x (torch.Tensor): 結果Tensor
        range (tuple[int, int]): 色範囲

    Returns:
        torch.Tensor: 結果画像Tensor
    """
    range_min, range_max = range
    x -= range_min
    if (range_max - range_min) > 0:
        x /= range_max - range_min
    return tensor_to_img(x)


def pil_to_b64encode(img_pil: Image) -> str:
    """PIL画像base64エンコード

    Args:
        img_pil (Image): PIL画像

    Returns:
        str: base64エンコード文字列
    """
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    img_b64encode = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_b64encode


def img_to_b64encode(x: np.ndarray) -> str:
    """画像base64エンコード

    Args:
        x (np.ndarray): 画像

    Returns:
        str: base64エンコード文字列
    """
    x = np.squeeze(x)
    img_pil = Image.fromarray((np.rint(x * 255)).astype(np.uint8))
    img_b64encode = pil_to_b64encode(img_pil)
    return img_b64encode


def setget_session_data(
    request: Request,
    form: PredictForm,
    train_dataset: StreamingDataset | None,
    test_dataset: StreamingDataset | None,
    force: bool,
) -> SessionData:
    """セッションで使用するデータ取得

    Args:
        request (Request): リクエスト
        form (PredictForm): 検査フォーム
        train_dataset (StreamingDataset): 正常データセット
        test_dataset (StreamingDataset): 検査データセット
        force (bool): セッションにかかわらずモデル再作成

    Returns:
        SessionData: セッションで使用するデータ
    """
    if "session_id" in request.session:
        session_id = request.session["session_id"]
    else:
        session_id = str(uuid.uuid4())
        request.session["session_id"] = session_id

    if not force and not form.hdn_value_changed and session_data.model:
        model = session_data.model
    else:
        assert train_dataset is not None and test_dataset is not None
        model = load_model(form.sel_method, form.sel_backbone)
        if form.chk_model:
            # モデル読込み
            model.load_buffer(form.file_model.file, test_dataset)
        else:
        model.fit(DataLoader(train_dataset))
        session_data.session_id = session_id
        session_data.model = model
        session_data.test_dataset = test_dataset
    return session_data


def get_result(
    form: PredictForm,
    session_data: SessionData,
    test_index: int,
) -> Dict[str, Any]:
    """結果取得

    Args:
        form (PredictForm): 検査フォーム
        session_data (SessionData): セッションで使用するデータ
        test_index (int): 検査インデックス

    Returns:
        Dict[str, Any]: 検査結果
    """

    pxl_lvl_anom_score = session_data.pxl_lvl_anom_score
    score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()

    pxl_lvl_anom_score_org = pxl_lvl_anom_score.clone()
    scores = (
        pxl_lvl_anom_score_org.to(torch.float32).detach().numpy().reshape(-1)
    )
    score_mode = stats.mode(scores, axis=None, keepdims=False).mode

    if form.chk_relative:
        color_range = score_range
    else:
        color_range = (form.hdn_color_min, form.hdn_color_max)

    sample, *_ = session_data.test_dataset[test_index]
    sample_img = tensor_to_img(sample, normalize=True)
    sample_b64encode = img_to_b64encode(sample_img)

    fmap_img = pred_to_img(pxl_lvl_anom_score.clone(), color_range)
    fmap_b64encode = img_to_b64encode(fmap_img)

    plt.imshow(sample_img)
    plt.imshow(fmap_img, cmap="jet", alpha=0.5)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
        dpi=60,
    )
    buf.seek(0)
    overlay_img = Image.open(buf)
    overlay_img_rgb = overlay_img.convert("RGB")
    overlay_img_b64encode = pil_to_b64encode(overlay_img_rgb)

    score_rate = (
        np.count_nonzero(form.rng_threshold <= scores) / np.size(scores) * 100
    )
    score_string = f"異常度：{score_rate:.2f}％"

    # エラー
    is_error = (
        form.num_threshold_rate <= score_rate
        or form.rng_threshold_min <= float(score_range[0])
        or form.rng_threshold_max <= float(score_range[1])
    )

    range_max = 200 if form.chk_relative else 255

    fig = plt.figure(figsize=(10, 2))
    n, bins, patches = plt.hist(scores, 100, (0, range_max))
    for patch in patches:
        if form.rng_threshold <= patch.get_x():
            patch.set_facecolor(mcolors.BASE_COLORS["r"])
    for bin_range in range(len(bins) - 1):
        if score_mode < bins[bin_range + 1]:
            plt.annotate(f"{score_mode:.0f}", (score_mode - 2, n[bin_range]))
            break
    plt.annotate(f"{score_range[0]:.0f}", (score_range[0] - 2, 100))
    plt.annotate(f"{score_range[1]:.0f}", (score_range[1] - 2, 100))
    ax = plt.gca()
    ax.set_ylim([0, 20000])

    img_hist = io.BytesIO()
    fig.savefig(img_hist, format="jpeg", bbox_inches="tight")
    img_hist.seek(0)

    img_hist_b64encode = base64.b64encode(img_hist.getvalue())

    plt.close()

    session_data.current_index = test_index

    return {
        "test_size": len(session_data.test_dataset),
        "sample_b64encode": sample_b64encode,
        "fmap_b64encode": fmap_b64encode,
        "overlay_img_b64encode": overlay_img_b64encode,
        "score_rate": score_rate,
        "score_string": score_string,
        "is_error": is_error,
        "img_hist_b64encode": img_hist_b64encode,
    }


def testing(
    form: PredictForm, session_data: SessionData, test_index: int
) -> Dict[str, Any]:
    """検査実行

    Args:
        form (PredictForm): 検査フォーム
        session_data (SessionData): セッションで使用するデータ
        test_index (int): 検査インデックス

    Returns:
        Dict[str, Any]: 検査結果
    """

    test_dataset = session_data.test_dataset
    sample, *_ = test_dataset[test_index]
    model = session_data.model
    _, pxl_lvl_anom_score = model.predict(sample.unsqueeze(0))
    session_data.pxl_lvl_anom_score = pxl_lvl_anom_score

    result = get_result(form, session_data, test_index)
    return result


app = FastAPI()
app.mount(path="/static", app=StaticFiles(directory="static"), name="static")
app.add_middleware(SessionMiddleware, secret_key="SECRET_KEY")
templates = Jinja2Templates(directory="templates/")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(
    request: Request,
    form: PredictForm = Depends(PredictForm.as_form),
) -> Dict[str, Any]:
    """検査実行

    Args:
        request (Request): リクエスト
        form (PredictForm, optional): 検査フォーム.
            Defaults to Depends(PredictForm.as_form).

    Returns:
        Dict[str, Any]: 検査結果
    """
    try:
    if not form.chk_sample:
            # サンプルデータ使用なし、モデル読込みなし
        # test dataset will contain 1 test image
        train_dataset = StreamingDataset()
        test_dataset = StreamingDataset()

        # train images
            if form.chk_model:
                # モデル読込みチェックあり
                train_b64encode_images = []
            else:
        for training_image in form.file_train:
            if training_image.size == 0:
                        return {
                            "message": "正常画像を3枚以上選択してください。",
                            "error": "file_train length < 3",
                        }
                    train_dataset.add_pil_image(
                        Image.open(training_image.file)
                    )
        train_b64encode_images = train_dataset.get_b64encode_images()

        # test image
        for test_image in form.file_test:
            if test_image.size == 0:
                    return {
                        "message": "検査画像を1枚以上選択してください。",
                        "error": "file_test length < 1",
                    }
            test_dataset.add_pil_image(Image.open(test_image.file))
    else:
        train_dataset, test_dataset = MVTecDataset(
                form.sel_mvtec
            ).get_datasets()

    session_data = setget_session_data(
        request, form, train_dataset, test_dataset, False
    )

    test_result = testing(form, session_data, 0)
    except Exception as err:
        return {
            "message": "エラーが発生しました。",
            "error": f"Unexpected {err=}, {type(err)=}",
        }
    return {
        "train_b64encode_images": train_b64encode_images,
        "test_result": test_result,
    }


@app.post("/change/")
async def change_condition(
    request: Request = None,
    form: PredictForm = Depends(PredictForm.as_form),
) -> Dict[str, Any]:
    """エラー判定変更

    Args:
        request (Request): リクエスト
        form (PredictForm, optional): 検査フォーム.
            Defaults to Depends(PredictForm.as_form).

    Returns:
        Dict[str, Any]: 検査結果
    """
    try:
    if (
        "session_id" not in request.session
        or session_data.session_id != request.session["session_id"]
    ):
        return await predict(request, form)
    session_data_l = setget_session_data(request, form, None, None, False)

        test_result = get_result(
            form, session_data_l, session_data.current_index
        )
    except Exception as err:
        return {
            "message": "エラーが発生しました。",
            "error": f"Unexpected {err=}, {type(err)=}",
        }
    return {
        "test_result": test_result,
    }


@app.post("/change/{test_index}")
async def change_condition_index(
    test_index: int,
    request: Request = None,
    form: PredictForm = Depends(PredictForm.as_form),
) -> Dict[str, Any]:
    """検査対象変更

    Args:
        test_index (int): 検査対象インデックス（0～）
        request (Request, optional): リクエスト. Defaults to None.
        form (PredictForm, optional): 検査フォーム.
            Defaults to Depends(PredictForm.as_form).

    Returns:
        Dict[str, Any]: 検査結果
    """
    try:
    if (
        "session_id" not in request.session
        or session_data.session_id != request.session["session_id"]
    ):
        return await predict(request, form)
    session_data_l = setget_session_data(request, form, None, None, False)
    test_result = testing(form, session_data_l, test_index)
    except Exception as err:
        return {
            "message": "エラーが発生しました。",
            "error": f"Unexpected {err=}, {type(err)=}",
        }
    return {
        "test_result": test_result,
    }


@app.get("/save/")
async def save(
    request: Request = None, form: PredictForm = Depends(PredictForm.as_form)
):
    """モデル保存

    Args:
        request (Request): リクエスト
        form (PredictForm, optional): 検査フォーム.
            Defaults to Depends(PredictForm.as_form).

    Returns:
        Dict[str, Any]: 検査結果
    """
    try:
        if (
            "session_id" not in request.session
            or session_data.session_id != request.session["session_id"]
        ):
            _ = await predict(request, form)
        session_data_l = setget_session_data(request, form, None, None, False)
        model = session_data_l.model
        buffer = model.get_buffer()
        buffer.seek(0)
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected {err=}, {type(err)=}",
        )

    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename='model.tar'"},
    )
