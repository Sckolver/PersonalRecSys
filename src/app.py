import asyncio, json, os, pickle
from typing import Any, Dict, List

import numpy as np
import orjson
import polars as pl
import uvloop
from aiohttp import web
from catboost import CatBoostRanker, Pool
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import load_npz

# ---------- fast helpers ------------------------------------------------- #

HEADERS = {"Content-Type": "application/json"}
jresp   = lambda obj, code=200: web.Response(
    body=orjson.dumps(obj), status=code, headers=HEADERS
)

def _parse_cookies(body: Dict[str, Any], key_type, known_set):
    raw = body.get("cookies")
    if not isinstance(raw, list) or not raw:
        raise ValueError("cookies must be non-empty list")

    try:
        cookies = [key_type(c) for c in raw]
    except Exception:
        raise ValueError("wrong cookie dtype")

    known = [c for c in cookies if c in known_set]
    return known

# ---------- ALS ---------------------------------------------------------- #

def _als_batch(
    model: AlternatingLeastSquares,
    mat,
    u2i: Dict[Any, int],
    i2n_arr: np.ndarray,
    users: List[Any],
    top_n: int,
):
    idx = np.fromiter((u2i[u] for u in users), dtype=np.int32)
    recs, scores = model.recommend(idx, mat[idx], N=top_n,
                                   filter_already_liked_items=True)
    recs_flat   = np.asarray(recs).ravel()
    scores_flat = np.asarray(scores).ravel()
    nodes_flat  = np.take(i2n_arr, recs_flat)
    cookies_rep = np.repeat(users, top_n)
    return cookies_rep, nodes_flat, scores_flat

# ---------- feature assembly -------------------------------------------- #

def _assemble_features(
    cookies: np.ndarray,
    nodes: np.ndarray,
    als_score: np.ndarray,
    cookie_f: Dict[Any, np.ndarray],
    node_f: Dict[Any, np.ndarray],
):
    # собираем numpy-матрицу признаков без дорогих join-ов
    n_samples = len(cookies)
    feats_cookie = np.vstack([cookie_f[c] for c in cookies])
    feats_node   = np.vstack([node_f[n]   for n in nodes])
    X = np.hstack([als_score.reshape(-1, 1), feats_cookie, feats_node])
    return X

# ---------- request handler --------------------------------------------- #

async def _recommend(req: web.Request, use_cached: bool) -> web.Response:
    try:
        body = await req.json(loads=orjson.loads)
    except ValueError:
        return jresp({"error": "invalid json"}, 400)

    try:
        cookies = _parse_cookies(body, req.app["cookie_type"], req.app["u2i"])
    except ValueError as e:
        return jresp({"error": str(e)}, 400)

    if not cookies:
        return jresp({"recommendations": {}})

    loop = asyncio.get_running_loop()
    # 1) ALS + i2n в отдельном потоке
    cookies_np, nodes_np, als_score = await loop.run_in_executor(
        None,
        _als_batch,
        req.app["als_model"],
        req.app["als_mat"],
        req.app["u2i"],
        req.app["i2n_arr"],
        cookies,
        req.app["als_N_cand"],
    )

    # 2) подбираем нужный набор фичей
    node_f_dict = req.app["sasrec_f"] if use_cached else req.app["node_f"]
    X = _assemble_features(
        cookies_np, nodes_np, als_score,
        req.app["cookie_f"], node_f_dict
    )

    # 3) CatBoost тоже в executor, чтобы не стопорить event-loop
    preds = await loop.run_in_executor(
        None,
        lambda: req.app["cb"].predict(Pool(X, group_id=cookies_np))
    )

    # 4) top-k
    order = np.lexsort((-preds, cookies_np))
    cookies_sorted, nodes_sorted = cookies_np[order], nodes_np[order]
    top_k = req.app["top_k"]
    out: Dict[str, List[Any]] = {str(c): [] for c in cookies}
    for c, n in zip(cookies_sorted, nodes_sorted):
        lst = out[str(c)]
        if len(lst) < top_k:
            lst.append(n)

    return jresp({"recommendations": out})

# ---------- aiohttp app -------------------------------------------------- #

async def init_app() -> web.Application:
    art = "/app/artifacts"
    app = web.Application()
    app.router.add_post("/recommend",               lambda r: _recommend(r, False))
    app.router.add_post("/recommend_cached_sasrec", lambda r: _recommend(r, True))

    app["als_model"] = AlternatingLeastSquares.load(os.path.join(art, "als_model.npz"))
    app["als_mat"]   = load_npz(os.path.join(art, "user_item_mat.npz"))

    with open(os.path.join(art, "u2i.pkl"), "rb") as f:
        app["u2i"] = pickle.load(f)
    with open(os.path.join(art, "i2n.pkl"), "rb") as f:
        app["i2n_arr"] = np.asarray(pickle.load(f))

    app["cookie_type"] = type(next(iter(app["u2i"].keys())))

    # фичи -> словари id → numpy
    app["cookie_f"] = {
        r["cookie"]: np.asarray(r)[1:]
        for r in pl.read_parquet(os.path.join(art, "cookie_f.parquet")).rows_named()
    }
    app["node_f"] = {
        r["node"]: np.asarray(r)[1:]
        for r in pl.read_parquet(os.path.join(art, "node_f.parquet")).rows_named()
    }
    # sasrec кеш сразу кладём в тот же формат
    app["sasrec_f"] = {
        r["node"]: np.asarray(r)[2:]  # cookie, node, ...
        for r in pl.read_parquet(os.path.join(art, "sasrec_cached.parquet")).rows_named()
    }

    cb = CatBoostRanker()
    cb.load_model(os.path.join(art, "catboost_model.cbm"))
    app["cb"] = cb

    app["als_N_cand"] = int(os.getenv("ALS_N_CAND", 339))
    app["top_k"]      = int(os.getenv("TOP_K", 40))
    return app

def main():
    uvloop.install()
    web.run_app(init_app(), port=int(os.getenv("PORT", 8080)))

if __name__ == "__main__":
    main()
