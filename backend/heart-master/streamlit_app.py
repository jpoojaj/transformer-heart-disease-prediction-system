"""Streamlit wrapper to host the existing Three.js heart viewer."""
from __future__ import annotations

import http.server
import socket
import socketserver
import threading
from functools import partial
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).parent.resolve()
STATIC_DIR = PROJECT_ROOT
_SERVER_KEY = "static_server"


def _start_static_server() -> int:
    """Start a background HTTP server that serves the project directory.

    Returns the port number that the server is bound to.
    """
    if _SERVER_KEY in st.session_state:
        return st.session_state[_SERVER_KEY]["port"]

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(STATIC_DIR))
    socketserver.TCPServer.allow_reuse_address = True

    # Try to bind to an available ephemeral port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_sock:
        temp_sock.bind(("127.0.0.1", 0))
        host, port = temp_sock.getsockname()

    httpd = socketserver.TCPServer((host, port), handler)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    st.session_state[_SERVER_KEY] = {"httpd": httpd, "thread": thread, "port": port}
    return port


def main():
    st.set_page_config(page_title="Interactive Heart Viewer", layout="wide")
    st.title("3D Heart Viewer")
    st.caption("This Streamlit wrapper serves the original WebGL experience inside an iframe.")

    try:
        port = _start_static_server()
    except OSError as exc:
        st.error(f"Failed to start static file server: {exc}")
        st.stop()

    iframe_url = f"http://127.0.0.1:{port}/index.html?controls=0"

    components.html(
        f"""
        <iframe src="{iframe_url}" style="width:100%; height:90vh; border:none;" allow="fullscreen"></iframe>
        """,
        height=800,
        scrolling=False,
    )

    with st.expander("How it works"):
        st.markdown(
            """
            - A lightweight Python HTTP server hosts the static Three.js assets from this project directory.
            - The viewer above is the original `index.html` rendered inside an iframe.
            - You can continue to modify the HTML/JS files, and the Streamlit app will reflect the changes on refresh.
            """
        )

    st.info(
        "Run `streamlit run streamlit_app.py` and keep this terminal session active while interacting with the viewer."
    )


if __name__ == "__main__":
    main()
