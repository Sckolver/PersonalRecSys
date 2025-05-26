import asyncio
from pprint import pprint

import httpx

URL = "http://localhost:8080/recommend"
TEST_COOKIES = [52564, "12345"]


async def main() -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        payload = {"cookies": TEST_COOKIES}
        resp = await client.post(URL, json=payload)
        resp.raise_for_status()  # поднимет исключение, если код != 2xx

        data = resp.json()
        print("Статус:", resp.status_code)
        pprint(data, width=120)


if __name__ == "__main__":
    asyncio.run(main())
