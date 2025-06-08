import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import asyncio
import pickle
from typing import Any

import numpy as np
import orjson
import polars as pl
import uvloop
from aiohttp import web
from catboost import CatBoostRanker, Pool
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import load_npz

HEADERS_JSON = {"Content-Type": "application/json"}


def jresp(obj: dict[str, Any], status: int = 200) -> web.Response:
    """Быстрый JSON-ответ через orjson (без лишней копии)."""
    return web.Response(body=orjson.dumps(obj), status=status, headers=HEADERS_JSON)


def parse_cookies(body: dict[str, Any], key_type, known_set) -> tuple[list[Any], str]:
    """
    Проверяем/преобразуем список cookies из запроса.
    Возвращает: (список найденных в модели cookies, текст ошибки | None)
    """
    cookies_raw = body.get("cookies")
    if not isinstance(cookies_raw, list) or not cookies_raw:
        return [], "cookies must be non-empty list"

    try:
        cookies = [key_type(c) for c in cookies_raw]
    except Exception:
        return [], "wrong cookie dtype"

    known = [c for c in cookies if c in known_set]
    return known, None


def als_batch(
    model: AlternatingLeastSquares,
    mat,
    u2i: dict[Any, int],
    i2n_arr: np.ndarray,
    users: list[Any],
    top_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Запрашиваем рекомендации ALS сразу пачкой и разворачиваем
    в плоские массивы для последующей обработки.
    """
    idx = np.fromiter((u2i[u] for u in users), dtype=np.int32, count=len(users))

    recs, scores = model.recommend(
        userid=idx,
        user_items=mat[idx],
        N=top_n,
        filter_already_liked_items=True,
    )

    recs_flat: np.ndarray = np.asarray(recs).ravel()
    cookies_rep = np.repeat(np.asarray(users), top_n)
    scores_flat = np.asarray(scores).ravel()
    nodes_flat: np.ndarray = np.take(i2n_arr, recs_flat)

    return cookies_rep, nodes_flat, scores_flat


def assemble_features(
    cookies: np.ndarray,
    nodes: np.ndarray,
    als_score: np.ndarray,
    cookie_f: dict[Any, np.ndarray],
    node_f: dict[Any, np.ndarray],
) -> np.ndarray:
    """
    Склейка всех признаков (als_score + cookie_f + node_f) в один NumPy-матрицу,
    минуя Pandas/Polars — быстро и без GIL.
    """
    n = len(cookies)
    cookie_stack = np.vstack([cookie_f[c] for c in cookies])
    node_stack = np.vstack([node_f[n_] for n_ in nodes])
    X = np.hstack([als_score.reshape(n, 1), cookie_stack, node_stack])
    return X


async def handle_recommend(req: web.Request, use_sasrec_cached: bool) -> web.Response:
    """
    Главный эндпоинт. `use_sasrec_cached=True` — вариант с дополнительными
    кандидатами из заранее вычисленного SASRec.
    """
    try:
        body = await req.json(loads=orjson.loads)
    except ValueError:
        return jresp({"error": "invalid json"}, 400)

    cookies, err = parse_cookies(body, req.app["cookie_type"], req.app["u2i"])
    if err:
        return jresp({"error": err}, 400)
    if not cookies:
        return jresp({"recommendations": {}})

    loop = asyncio.get_running_loop()

    cookies_np, nodes_np, als_score = await loop.run_in_executor(
        None,
        als_batch,
        req.app["als_model"],
        req.app["als_mat"],
        req.app["u2i"],
        req.app["i2n_arr"],
        cookies,
        req.app["als_N_cand"],
    )

    node_features = req.app["node_f"]
    X = assemble_features(
        cookies_np,
        nodes_np,
        als_score,
        req.app["cookie_f"],
        node_features,
    )

    preds = await loop.run_in_executor(
        None,
        lambda: req.app["cb"].predict(Pool(X, group_id=cookies_np)),
    )

    order = np.lexsort((-preds, cookies_np))
    cookies_sorted = cookies_np[order]
    nodes_sorted = nodes_np[order]
    top_k = req.app["top_k"]

    result: dict[str, list[Any]] = {str(c): [] for c in cookies}
    for c, n in zip(cookies_sorted, nodes_sorted):
        lst = result[str(c)]
        if len(lst) < top_k:
            lst.append(n)

    return jresp({"recommendations": result})


async def init_app() -> web.Application:
    """Инициализация aiohttp-приложения и загрузка артефактов."""
    art = "/app/artifacts"
    app = web.Application()
    app.router.add_post("/recommend", lambda r: handle_recommend(r, False))
    app.router.add_post(
        "/recommend_cached_sasrec",
        lambda r: handle_recommend(r, True),
    )

    app["als_model"] = AlternatingLeastSquares.load(os.path.join(art, "als_model.npz"))
    app["als_mat"] = load_npz(os.path.join(art, "user_item_mat.npz"))

    with open(os.path.join(art, "u2i.pkl"), "rb") as f:
        app["u2i"]: dict[Any, int] = pickle.load(f)
    with open(os.path.join(art, "i2n.pkl"), "rb") as f:
        i2n_dict: dict[int, Any] = pickle.load(f)

    max_idx = max(i2n_dict)
    i2n_arr = np.empty(max_idx + 1, dtype=object)
    for idx, node in i2n_dict.items():
        i2n_arr[idx] = node
    app["i2n_arr"] = i2n_arr

    app["cookie_type"] = type(next(iter(app["u2i"].keys())))

    def parquet_to_dict(path: str, key_col: str) -> dict[Any, np.ndarray]:
        df = pl.read_parquet(path)
        return {row[0]: np.asarray(row[1:], dtype=np.float32) for row in df.rows()}

    app["cookie_f"] = parquet_to_dict(os.path.join(art, "cookie_f.parquet"), "cookie")
    app["node_f"] = parquet_to_dict(os.path.join(art, "node_f.parquet"), "node")

    cb = CatBoostRanker()
    cb.load_model(os.path.join(art, "catboost_model.cbm"))
    app["cb"] = cb

    app["als_N_cand"] = int(os.getenv("ALS_N_CAND", 339))
    app["top_k"] = int(os.getenv("TOP_K", 40))

    return app


def main() -> None:
    """Точка входа."""
    uvloop.install()
    web.run_app(init_app(), port=int(os.getenv("PORT", 8080)))


if __name__ == "__main__":
    main()
