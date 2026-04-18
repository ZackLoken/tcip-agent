"""Entry point: python -m tcip_web"""

import uvicorn


def main():
    uvicorn.run("tcip_web.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
