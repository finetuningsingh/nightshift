"""Driver for the ONE bounded reference-experiment rental (owner authorization 2026-11-02: $3 extra, two hours, whichever comes first).

    python3 scripts/reference_run/rental.py init|create|ready|show|delete

The Thunder token is read from the owner's CLI login into this process's environment and never printed or written. The rental ledger
(`REF_LEDGER_DIR/rentals.json`) carries its own $3.00 hard stop; `create` refuses a cap that would exceed it, arms the detached watchdog
first, and tags the instance so every delete is ownership-checked. Only the one session named here is ever touched.
"""
import json, os, socket, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["THUNDER_API_KEY"] = json.load(open(os.path.expanduser("~/.thunder/cli_config.json")))["token"]
from nightshift.lab.contracts import Shape
from nightshift.lab.gpu_rentals import RentalManager
from nightshift.lab.gpu_thunder import ThunderProvider

ROOT = Path(os.environ.get("REF_LEDGER_DIR", "/Users/ankitsingh/glimmer-lab-live/ref-rentals"))
SESSION = "ref-1"
HARD_STOP_USD = 3.0
CAP_HOURS = 2.0
prov = ThunderProvider("THUNDER_API_KEY")  # default type map: a100 -> a100xl
mgr = RentalManager(ROOT / "rentals.json", prov, clock=time.time, machine=socket.gethostname(),
                    watchdog_provider_args=["--provider", "thunder", "--token-env", "THUNDER_API_KEY"])
cmd = sys.argv[1]
if cmd == "init":
    ROOT.mkdir(parents=True, exist_ok=True)
    mgr.init(HARD_STOP_USD, socket.gethostname(), "owner 2026-11-02: reference experiment, $3 extra and a two-hour stop, whichever comes first")
    print("init ok")
elif cmd == "create":
    r = mgr.create(SESSION, Shape("a100", 1, 8, 200, "base"), CAP_HOURS, "reference", SESSION)
    print("created", r.session, "handle", r.handle, "cap_epoch", r.cap_epoch)
elif cmd == "ready":
    r = mgr.rental(SESSION); inst = mgr.wait_ready(r, 900); t = mgr.endpoint(r)
    print("ready", inst.handle, inst.status, t.host, t.port, t.user)
elif cmd == "show":
    print(json.dumps(mgr.read(), indent=1)[:3500]); print(json.dumps(mgr.cost()))
elif cmd == "delete":
    mgr.delete(SESSION, sys.argv[2] if len(sys.argv) > 2 else "operator"); print("deleted")
