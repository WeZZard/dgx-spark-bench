"""Exercise the load generator against a mock OpenAI-compatible SSE endpoint.

Run it before spending server time on a harness change:

    python3 tests/test-load-against-mock.py

It needs `requests`, so it runs on the nodes rather than on a laptop.
Checks the parts of `sparkbench measure` that do not need a real engine.

Validates the parts of `measure` that do not need a real engine: SSE framing,
usage-based token counting, concurrency, needle detection, and that served and
decode rates come out where hand-calculation says they should.
"""
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sparkbench import load, rates, workload

N_TOKENS = 20
GAP = 0.02          # 20 ms between tokens; below this, TCP coalescing
                    # dominates and the mock stops being a clock
PREFILL = 0.30      # 300 ms before the first token


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        # echo any BUILD-KEY so the needle check has something to find
        key = ""
        if "BUILD-KEY-" in prompt:
            i = prompt.index("BUILD-KEY-")
            key = prompt[i:i + 22].split()[0].rstrip(".")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(PREFILL)
        for i in range(N_TOKENS):
            piece = (key + " ") if (i == 0 and key) else f"tok{i} "
            ev = {"choices": [{"delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
            time.sleep(GAP)
        usage = {"choices": [], "usage": {"completion_tokens": N_TOKENS,
                                          "prompt_tokens": 1234}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


# ThreadingHTTPServer, not HTTPServer: the single-threaded one serialises
# requests, so a 4-user run takes 4x as long and aggregate served rate comes
# out equal to single-stream. That is the harness reporting a serialising
# server correctly, but it tests the mock rather than the code.
srv = ThreadingHTTPServer(("127.0.0.1", 38999), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
time.sleep(0.3)

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")

# --- one request -----------------------------------------------------------
ps = workload.build("agentic-4k", 1)
r = load.run_load("http://127.0.0.1:38999", "m", ps, users=1, max_tokens=32)
t = r.traces[0]
check("request succeeded", t.ok, f"(err={t.error})")
check("token count came from usage", t.output_tokens == N_TOKENS, f"got {t.output_tokens}")
check("prompt tokens read back", t.input_tokens == 1234, f"got {t.input_tokens}")
check("usage was reported", r.usage_reported)
check("needle found", r.needle_hits == 1, f"{r.needle_hits}/{r.needle_total}")

served = rates.served(r.traces)
decode = rates.decode(r.traces)
# wall ~ PREFILL + N*GAP; decode span ~ (N-1)*GAP
print(f"  INFO  served={served.value:.1f} decode={decode.value:.1f} wall={served.wall_s:.3f}s")
check("served rate is charged for prefill", served.value < decode.value,
      f"({served.value:.1f} < {decode.value:.1f})")
# Loose bound: the mock's sleep is not a precise clock and chunks coalesce,
# so this checks the order of magnitude and the direction, not the value.
check("decode rate is in the right decade", 1/GAP * 0.5 < decode.value < 1/GAP * 3,
      f"got {decode.value:.1f}, mock ideal {1/GAP:.0f}")
check("ttft includes prefill", t.ttft_ms and t.ttft_ms >= PREFILL * 1000,
      f"got {t.ttft_ms:.0f} ms")

# --- concurrency -----------------------------------------------------------
ps4 = workload.build("agentic-4k", 4)
r4 = load.run_load("http://127.0.0.1:38999", "m", ps4, users=4, max_tokens=32)
s4 = rates.served(r4.traces)
check("4 concurrent requests all ok", all(x.ok for x in r4.traces))
check("aggregate served rate beats single", s4.value > served.value * 1.5,
      f"({s4.value:.1f} vs {served.value:.1f})")
check("all 4 needles found", r4.needle_hits == 4, f"{r4.needle_hits}/4")
print(f"  INFO  c4 served={s4.value:.1f} over {s4.wall_s:.3f}s wall, "
      f"{s4.output_tokens} tokens")

print("\nmock harness test:", "all passed" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
