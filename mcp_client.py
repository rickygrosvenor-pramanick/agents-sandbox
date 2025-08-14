import subprocess
import json
import sys
import threading

def _stream_stderr(proc):
    for line in iter(proc.stderr.readline, b""):
        sys.stderr.write("[server] " + line.decode(errors="replace"))

def send_json(proc, obj):
    """Send one JSON message. FastMCP stdio expects line-delimited JSON."""
    proc.stdin.write((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
    proc.stdin.flush()

def read_json(proc):
    """
    Read one message. Prefer line-delimited JSON, but gracefully handle
    Content-Length framing if the server ever uses it.
    """
    # Try line-delimited JSON first
    line = proc.stdout.readline()
    if not line:
        return None
    s = line.decode("utf-8", errors="ignore").strip()
    if s:
        # If this already looks like JSON, parse it
        if s[0] in "{[":
            return json.loads(s)
        # If this is a Content-Length header, fall back to LSP framing
        if s.lower().startswith("content-length"):
            # Read blank line
            blank = proc.stdout.readline()
            # Parse length
            try:
                length = int(s.split(":", 1)[1].strip())
            except Exception:
                length = 0
            body = proc.stdout.read(length).decode("utf-8", errors="ignore")
            return json.loads(body)
    # If we got here, keep reading until we get JSON
    while True:
        line = proc.stdout.readline()
        if not line:
            continue
        s = line.decode("utf-8", errors="ignore").strip()
        if not s:
            continue
        if s.lower().startswith("content-length"):
            _ = proc.stdout.readline()  # consume blank line
            length = int(s.split(":", 1)[1].strip())
            body = proc.stdout.read(length).decode("utf-8", errors="ignore")
            return json.loads(body)
        if s[0] in "{[":
            return json.loads(s)

if __name__ == "__main__":
    # Launch the server as a subprocess
    proc = subprocess.Popen(
        [sys.executable, "-u", "mcp_rag_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    # Stream server logs so you can see what it's doing
    threading.Thread(target=_stream_stderr, args=(proc,), daemon=True).start()

    # 1) initialize
    send_json(proc, {
        "jsonrpc":"2.0",
        "id":1,
        "method":"initialize",
        "params":{
            "protocolVersion":"2024-11-05",
            "capabilities":{},
            "clientInfo":{"name":"test-client","version":"0.1"}
        }
    })
    init_resp = read_json(proc)
    print("Initialize response:", init_resp)

    # 2) notifications/initialized
    send_json(proc, {
        "jsonrpc":"2.0",
        "method":"notifications/initialized",
        "params":{}
    })

    # 3) tools/list
    send_json(proc, {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})
    tools_resp = read_json(proc)
    print("Tools list:", tools_resp)

    # 4) tools/call
    send_json(proc, {
        "jsonrpc":"2.0",
        "id":3,
        "method":"tools/call",
        "params":{
            "name":"business_analyst_story_generator",
            "arguments":{"prompt":"Reduce cost/income ratio from 63% to 47.8%"}
        }
    })
    call_resp = read_json(proc)
    print("Tool call response:", call_resp)
