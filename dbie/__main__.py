"""Entry point: python -m dbie"""

import uvicorn

from dbie import config

if __name__ == "__main__":
    loop_setting = "uvloop" if config.USE_UVLOOP else "asyncio"
    uvicorn.run(
        "dbie.server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        workers=1,  # Single worker — multiple workers destroy batching efficiency
        loop=loop_setting,
        http="httptools",
        log_level="info",
    )
