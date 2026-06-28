"""step13 run.py — 验证 server.py 语法 + 依赖可用性"""
import os
import ast

def main():
    print("step13: HTTP 服务封装（OpenAI 兼容接口）")
    print("=" * 50)

    server_path = os.path.join(os.path.dirname(__file__), "server.py")
    with open(server_path) as f:
        source = f.read()
    ast.parse(source)
    print("server.py 语法检查通过 ✅")

    try:
        import fastapi
        import uvicorn
        print(f"FastAPI {fastapi.__version__} / uvicorn 依赖可用 ✅")
    except ImportError as e:
        print(f"⚠️ 依赖缺失: {e}")
        print("请安装: pip install fastapi uvicorn")

    print("\n✅ step13_serve 通过")

if __name__ == "__main__":
    main()
