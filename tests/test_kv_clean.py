"""Self-check for scripts/ds4-kv-clean against a temp KV root (never ~/.ds4)."""

import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import struct
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_loader = importlib.machinery.SourceFileLoader(
    "ds4_kv_clean", os.path.join(HERE, "..", "scripts", "ds4-kv-clean"))
_spec = importlib.util.spec_from_loader("ds4_kv_clean", _loader)
kvclean = importlib.util.module_from_spec(_spec)
_loader.exec_module(kvclean)

DAY = 86400
SHA = "0123456789abcdef0123456789abcdef0123456"


def checkpoint(dirpath, n, age_days, size=4096):
    """A <sha>.kv whose header last_used and mtime are age_days old."""
    t = int(time.time() - age_days * DAY)
    path = os.path.join(dirpath, f"{SHA}{n}.kv")
    header = b"KVC" + bytes(21) + struct.pack("<QQQ", t, t, 0)
    with open(path, "wb") as fp:
        fp.write(header + bytes(size - len(header)))
    os.utime(path, (t, t))
    return path


class KvCleanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = self.tmp.name
        self.root = os.path.join(base, "server-kv")
        self.state = os.path.join(base, "state")
        launchers = os.path.join(base, "launchers")
        os.makedirs(launchers)
        with open(os.path.join(launchers, "run-x.sh"), "w") as fp:
            fp.write('KV_DIR="$HOME/.ds4/server-kv/model-a"\nKV_DEFAULT=model-live\n')
        for name in ("model-a", "model-live", "bench-scratch", "orphan-old", "orphan-new"):
            os.makedirs(os.path.join(self.root, name))
        d = lambda n: os.path.join(self.root, n)
        self.a_old = checkpoint(d("model-a"), 1, 30)
        self.a_new = checkpoint(d("model-a"), 2, 1)
        self.tmpfile = os.path.join(d("model-a"), f"{SHA}3.kv.tmp.999999")
        open(self.tmpfile, "wb").close()
        self.live_old = checkpoint(d("model-live"), 4, 30)
        checkpoint(d("bench-scratch"), 5, 0)
        checkpoint(d("orphan-old"), 6, 30)
        old = time.time() - 30 * DAY
        os.utime(d("orphan-old"), (old, old))
        self.orphan_new = checkpoint(d("orphan-new"), 7, 1)
        self.argv = ["--kv-root", self.root, "--state-dir", self.state,
                     "--launcher-dir", launchers, "--live-dir", d("model-live"),
                     "--min-free-gb", "0", "--json"]

    def tearDown(self):
        self.tmp.cleanup()

    def run_clean(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = kvclean.main(self.argv + list(extra))
        return rc, json.loads(out.getvalue())

    def test_rules(self):
        rc, rep = self.run_clean()
        self.assertEqual(rc, 0, rep["errors"])
        self.assertFalse(os.path.exists(self.a_old))                              # (b) old
        self.assertTrue(os.path.exists(self.a_new))                               # (b) fresh
        self.assertFalse(os.path.exists(self.tmpfile))                            # (a) dead-pid tmp
        self.assertFalse(os.path.exists(os.path.join(self.root, "bench-scratch")))  # (a) scratch
        self.assertTrue(os.path.exists(self.live_old))                            # (e) live dir untouched
        self.assertFalse(os.path.exists(os.path.join(self.root, "orphan-old")))   # (c) old orphan
        self.assertTrue(os.path.exists(self.orphan_new))                          # (c) recent orphan
        self.assertIn("orphan-dir", [w["code"] for w in rep["warnings"]])
        self.assertEqual(rep["dirs_deleted"], 2)
        self.assertEqual(rep["skipped"], 1)
        self.assertTrue(rep["per_dir"]["model-live"]["live"])
        with open(os.path.join(self.state, "kv-clean-last.json")) as fp:
            self.assertEqual(json.load(fp)["freed_bytes"], rep["freed_bytes"])
        with open(os.path.join(self.state, "kv-clean.log")) as fp:
            self.assertRegex(fp.read(), r"\] CLEAN done: freed \d+\.\d\d GiB, skipped=1, errors=0")

    def test_dry_run_deletes_nothing(self):
        before = sorted(os.walk(self.root))
        rc, rep = self.run_clean("--dry-run")
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(os.walk(self.root)), before)
        self.assertGreater(rep["freed_bytes"], 0)
        self.assertTrue(rep["dry_run"])
        with open(os.path.join(self.state, "kv-clean.log")) as fp:
            self.assertRegex(fp.read(), r"\] DRY-RUN done: would free \d+\.\d\d GiB, skipped=1, errors=0")

    def test_min_free_evicts_oldest_non_live(self):
        rc, rep = self.run_clean("--min-free-gb", "1e9", "--max-age-days", "3650")
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(self.a_old))
        self.assertTrue(os.path.exists(self.live_old))
        self.assertIn("disk-low", [w["code"] for w in rep["warnings"]])


if __name__ == "__main__":
    unittest.main()
