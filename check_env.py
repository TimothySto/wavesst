import sys
print("Python:", sys.version)

checks = [
    ("numpy", lambda: __import__("numpy").__version__),
    ("cython", lambda: __import__("cython").__version__),
    ("torch", lambda: __import__("torch").__version__),
    ("torch.cuda", lambda: "available" if __import__("torch").cuda.is_available() else "NOT available"),
    ("cupy", lambda: __import__("cupy").__version__),
    ("scipy", lambda: __import__("scipy").__version__),
    ("ssqueezepy", lambda: __import__("ssqueezepy").__version__),
    ("pywt", lambda: __import__("pywt").__version__),
    ("pytest", lambda: __import__("pytest").__version__),
    ("hypothesis", lambda: __import__("hypothesis").__version__),
    ("numba", lambda: __import__("numba").__version__),
    ("vispy", lambda: __import__("vispy").__version__),
    ("OpenGL", lambda: __import__("OpenGL").__version__),
]

for name, get_ver in checks:
    try:
        ver = get_ver()
        print(f"  OK  {name}: {ver}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
