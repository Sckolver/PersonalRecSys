import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import orjson
import polars as pl
import uvloop
from aiohttp import web
from catboost import CatBoostRanker, Pool
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import load_npz


def recommend_als(
    model: AlternatingLeastSquares,
    mat,
    u2i: Dict[Any, int],
    i2n: Dict[int, Any],
    users: List[Any],
    n_cand: int,
) -> pl.DataFrame:
    idx = [u2i[u] for u in users]
    recs, scores = model.recommend(
        userid=idx, user_items=mat[idx], N=n_cand, filter_already_liked_items=True
    )
    return pl.DataFrame(
        {
            "cookie": np.repeat(users, n_cand),
            "node": np.concatenate([[i2n[j] for j in r] for r in recs]),
            "als_score": np.concatenate(scores),
        }
    )


async def handle_recommend(request: web.Request) -> web.Response:
    try:
        body = await request.json(loads=orjson.loads)
    except ValueError:
        return web.json_response({"error": "invalid json"}, status=400)

    cookies = body.get("cookies")
    if not isinstance(cookies, list) or not cookies:
        return web.json_response(
            {"error": "cookies must be non-empty list"}, status=400
        )

    key_type = request.app["cookie_type"]
    try:
        cookies = [key_type(c) for c in cookies]
    except (ValueError, TypeError):
        pass

    u2i: Dict[Any, int] = request.app["u2i"]
    known = [c for c in cookies if c in u2i]
    if not known:
        return web.json_response({"recommendations": {}})

    df_cand = recommend_als(
        model=request.app["als_model"],
        mat=request.app["als_mat"],
        u2i=u2i,
        i2n=request.app["i2n"],
        users=known,
        n_cand=request.app["als_N_cand"],
    )

    df_cand = (
        df_cand.join(request.app["cookie_f"], on="cookie", how="left")
        .join(request.app["node_f"], on="node", how="left")
        .fill_null(0)
    )

    feats = request.app["feats"]
    df_pd = df_cand.select(feats).to_pandas()
    pool = Pool(df_pd.drop(columns=["cookie"]), group_id=df_pd["cookie"].to_numpy())

    scores = request.app["cb_ranker"].predict(pool)
    df_cand = df_cand.with_columns(pl.Series(scores).alias("score"))

    top_k = request.app["top_k"]
    df_top = (
        df_cand.sort(["cookie", "score"], descending=[False, True])
        .group_by("cookie")
        .head(top_k)
    )

    result = {
        str(c): df_top.filter(pl.col("cookie") == c)["node"].to_list() for c in known
    }

    return web.json_response({"recommendations": result})


async def init_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/recommend", handle_recommend)

    artifacts = os.getenv("ARTIFACTS_DIR", "artifacts")

    app["als_model"] = AlternatingLeastSquares.load(
        os.path.join(artifacts, "als_model.npz")
    )
    app["als_mat"] = load_npz(os.path.join(artifacts, "user_item_mat.npz"))

    with open(os.path.join(artifacts, "u2i.pkl"), "rb") as f:
        app["u2i"] = pickle.load(f)
    with open(os.path.join(artifacts, "i2n.pkl"), "rb") as f:
        app["i2n"] = pickle.load(f)

    app["cookie_type"] = type(next(iter(app["u2i"].keys())))

    cb = CatBoostRanker()
    cb.load_model(os.path.join(artifacts, "catboost_model.cbm"))
    app["cb_ranker"] = cb

    app["cookie_f"] = pl.read_parquet(os.path.join(artifacts, "cookie_f.parquet"))
    app["node_f"] = pl.read_parquet(os.path.join(artifacts, "node_f.parquet"))

    with open(os.path.join(artifacts, "feats.json"), "r") as f:
        app["feats"] = json.load(f)

    app["als_N_cand"] = int(os.getenv("ALS_N_CAND", 339))
    app["top_k"] = int(os.getenv("TOP_K", 40))

    return app


def main() -> None:
    web.run_app(init_app(), port=int(os.getenv("PORT", 8080)))


if __name__ == "__main__":
    uvloop.install()
    main()
