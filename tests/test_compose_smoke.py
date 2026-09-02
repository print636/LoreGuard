import unittest
from unittest.mock import patch

from scripts.run_compose_smoke import wait_until_ready


class ComposeSmokeTests(unittest.TestCase):
    def test_readiness_retries_connection_reset_during_container_startup(self):
        with (
            patch(
                "scripts.run_compose_smoke.request_json",
                side_effect=[ConnectionResetError("starting"), {"status": "ok"}],
            ) as request,
            patch("scripts.run_compose_smoke.time.sleep"),
        ):
            wait_until_ready("http://127.0.0.1:8000", 5)
        self.assertEqual(2, request.call_count)


if __name__ == "__main__":
    unittest.main()
