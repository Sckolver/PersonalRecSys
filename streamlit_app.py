from typing import List
import streamlit as st
import httpx
import asyncio
import json

RECOMMEND_URL = "http://localhost:8080/recommend_cached_sasrec"


def parse_cookies(raw: str) -> List[str]:
    for sep in [",", ";"]:
        raw = raw.replace(sep, " ")
    parts = raw.split()
    return [p.strip() for p in parts if p.strip()]


async def get_recommendations(cookies: List[str]) -> dict:
    payload = {"cookies": cookies}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(RECOMMEND_URL, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


def render_recommendations(res: dict):
    if res.get("error"):
        st.error(f"Ошибка: {res['error']}")
        return

    recs = res.get("recommendations", {})
    if not recs:
        st.info("В списке есть невалидные cookie.")
        return

    st.markdown(
        """
        <div class="scroll-box">
        """,
        unsafe_allow_html=True,
    )

    for cookie, nodes in recs.items():
        if not nodes:
            continue

        with st.expander(f"Cookie: {cookie}", expanded=False):
            if nodes:
                for idx, node in enumerate(nodes, start=1):
                    st.write(f"{idx}. {node}")
            else:
                st.write("— для этой cookie нет рекомендаций.")

    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .scroll-box {
            padding-right: 10px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            background-color: #fafafa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    if "result" not in st.session_state:
        st.session_state["result"] = None

    st.set_page_config(page_title="MARIA Personal RecSys")
    st.title("MARIA Personal RecSys")
    st.write("Введите список cookie (через запятую, точку с запятой или пробел)")

    raw_input = st.text_area(
        label="Список cookie",
        height=100,
        placeholder="52564, 12345, my_cookie",
        label_visibility="collapsed",
    )

    col1, _, col3 = st.columns([4, 4.9, 2.1])

    with col1:
        if st.button("Получить рекомендации"):
            cookies = parse_cookies(raw_input)
            if not cookies:
                st.warning("Нужно указать хотя бы один валидный cookie.")
            else:
                with st.spinner("Получение рекомендаций"):
                    result = asyncio.run(get_recommendations(cookies))
                st.session_state["result"] = result

    with col3:
        result = st.session_state.get("result")
        if result and result.get("recommendations"):
            json_str = json.dumps(
                {"recommendations": result["recommendations"]},
                ensure_ascii=False,
                indent=4,
            )
            st.download_button(
                label="Скачать JSON",
                data=json_str,
                file_name="recommendations.json",
                mime="application/json",
            )

    if st.session_state.get("result") is not None:
        render_recommendations(st.session_state["result"])


if __name__ == "__main__":
    main()
