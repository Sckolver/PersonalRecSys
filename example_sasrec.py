import aiohttp
import asyncio
import orjson


async def get_sasrec_recommendations(url: str, cookies: list):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{url}/recommend_cached_sasrec",
            json={"cookies": cookies},
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                data = await response.json(loads=orjson.loads)
                print("Recommendations:", data)
            else:
                print("Error:", response.status)
                print(await response.text())


if __name__ == "__main__":
    url = "http://localhost:8080"
    cookies = [52564, "11111"]

    asyncio.run(get_sasrec_recommendations(url, cookies))
