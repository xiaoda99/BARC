#!/usr/bin/env python
import sys
import jupyter_client

# if len(sys.argv) < 3:
#     print("Usage: exec_on_kernel.py <kernel-id-or-connection-file> <code>")
#     sys.exit(1)
# target = sys.argv[1]      # e.g. 18562 or kernel-18562.json or full path
# code   = sys.argv[2]      # e.g. 'print("hi"); 1+2'

if len(sys.argv) < 2:
    print("Usage: exec_on_kernel.py <code>")
    sys.exit(1)
target = '0c622c10-daf0-42f0-afe0-d6e8505267e1'
code   = sys.argv[1]      # e.g. 'print("hi"); 1+2'


# Resolve to the actual JSON file, e.g. ~/.local/share/jupyter/runtime/kernel-18562.json
cf = jupyter_client.find_connection_file(target)

kc = jupyter_client.BlockingKernelClient(connection_file=cf)
kc.load_connection_file()
kc.start_channels()

def print_output(msg):
    """Handle each IOPub message emitted by the kernel."""
    msg_type = msg["header"]["msg_type"]
    content  = msg["content"]

    if msg_type == "stream" and content.get("name") == "stdout":
        # prints
        sys.stdout.write(content.get("text", ""))
        sys.stdout.flush()

    elif msg_type in ("execute_result", "display_data"):
        data = content.get("data", {})
        text = data.get("text/plain", "")
        if text:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

    elif msg_type == "error":
        # simple traceback printing
        sys.stderr.write("".join(content.get("traceback", [])) + "\n")
        sys.stderr.flush()

# This helper sends the execute_request *and* drains IOPub, calling our hook. :contentReference[oaicite:3]{index=3}
kc.execute_interactive(
    code,
    output_hook=print_output,
    timeout=30,
    stop_on_error=True,
)

kc.stop_channels()
