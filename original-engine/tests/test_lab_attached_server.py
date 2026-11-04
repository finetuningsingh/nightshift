import unittest

from nightshift.lab.attached_server import AttachedModelServer
from nightshift.lab.contracts import AdapterRef, AdapterFile, BaseModelRef, ScorerError, ServeSpec

BASE = BaseModelRef("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68")


def adapter() -> AdapterRef:
    return AdapterRef.from_files("ref1", [AdapterFile("adapter_config.json", "a" * 64, 10), AdapterFile("adapter_model.safetensors", "b" * 64, 20)])


class AttachedModelServerTest(unittest.TestCase):
    def server(self, models, clock):
        names = {adapter().sha256: "ref"}
        return AttachedModelServer("http://127.0.0.1:1/", "muse-glimmer", names, rental_id="r1", rate_usd_per_hour=1.2, clock=clock, list_models=lambda url: models)

    def test_serves_base_and_adapter_and_reports_window_cost(self):
        t = [1000.0]
        s = self.server(["muse-glimmer", "ref"], lambda: t[0])
        session = s.serve(ServeSpec(BASE, (adapter(),), 8192, 60.0, "score", None))
        self.assertEqual(session.endpoint.model_for(adapter()), "ref")
        self.assertEqual(session.endpoint.model_for(None), "muse-glimmer")
        t[0] += 1800
        usage = session.close()
        self.assertEqual((usage.minutes, usage.usd, usage.rental_ids), (30.0, 0.6, ("r1",)))
        self.assertEqual(session.close(), usage)

    def test_refuses_when_the_server_does_not_list_the_adapter(self):
        s = self.server(["muse-glimmer"], lambda: 0.0)
        with self.assertRaises(ScorerError):
            s.serve(ServeSpec(BASE, (adapter(),), 8192, 60.0, "score", None))

    def test_refuses_an_adapter_it_was_not_given(self):
        s = AttachedModelServer("http://x", "muse-glimmer", {}, rental_id="r1", rate_usd_per_hour=1.0, list_models=lambda u: ["muse-glimmer"])
        with self.assertRaises(ScorerError):
            s.serve(ServeSpec(BASE, (adapter(),), 8192, 60.0, "score", None))


if __name__ == "__main__":
    unittest.main()
