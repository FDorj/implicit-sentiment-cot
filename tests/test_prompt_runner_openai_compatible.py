import os
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import requests

from src.prompt_runner import OpenAICompatiblePromptRunner, PromptRunner
from src.experiment_config import get_model_id


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} error", response=self)
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, json, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "json": copy.deepcopy(json),
                "headers": headers,
                "timeout": timeout,
            }
        )
        return FakeResponse(self.payload)


class SequenceSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "json": copy.deepcopy(json),
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self.responses.pop(0)


class ExceptionThenResponseSession:
    def __init__(self, exceptions, response):
        self.exceptions = list(exceptions)
        self.response = response
        self.calls = []

    def post(self, url, json, headers, timeout):
        self.calls.append(
            {
                "url": url,
                "json": copy.deepcopy(json),
                "headers": headers,
                "timeout": timeout,
            }
        )
        if self.exceptions:
            raise self.exceptions.pop(0)
        return self.response


class OpenAICompatiblePromptRunnerTests(unittest.TestCase):
    def test_posts_chat_completion_payload_and_parses_text(self):
        session = FakeSession(
            {
                "choices": [
                    {
                        "message": {
                            "content": " positive\n",
                        }
                    }
                ]
            }
        )

        runner = OpenAICompatiblePromptRunner(
            model="Gemini-2.5-Flash",
            base_url="https://gateway.example/v1/",
            api_key="secret",
            session=session,
        )

        output = runner.run("Classify this.", temperature=0.2, max_tokens=7)

        self.assertEqual(output, "positive")
        self.assertEqual(len(session.calls), 1)
        call = session.calls[0]
        self.assertEqual(call["url"], "https://gateway.example/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(call["headers"]["Content-Type"], "application/json")
        self.assertEqual(call["timeout"], 300)
        self.assertEqual(
            call["json"],
            {
                "model": "Gemini-2.5-Flash",
                "messages": [{"role": "user", "content": "Classify this."}],
                "temperature": 0.2,
                "max_tokens": 7,
                "stream": False,
            },
        )

    def test_can_floor_max_tokens_for_reasoning_compatible_models(self):
        session = FakeSession({"choices": [{"message": {"content": "negative"}}]})
        env = {"OPENAI_COMPAT_MIN_MAX_TOKENS": "64"}

        with patch.dict(os.environ, env, clear=False):
            runner = OpenAICompatiblePromptRunner(
                model="Gemini-2.5-Flash",
                base_url="https://gateway.example/v1",
                api_key="",
                session=session,
            )
            output = runner.run("Classify.", temperature=0.0, max_tokens=16)

        self.assertEqual(output, "negative")
        self.assertEqual(session.calls[0]["json"]["max_tokens"], 64)

    def test_retries_transient_gateway_errors(self):
        session = SequenceSession(
            [
                FakeResponse({"error": "bad gateway"}, status_code=502),
                FakeResponse({"choices": [{"message": {"content": "neutral"}}]}),
            ]
        )

        runner = OpenAICompatiblePromptRunner(
            model="Gemini-2.5-Flash",
            base_url="https://gateway.example/v1",
            api_key="",
            session=session,
        )
        runner.retry_sleep_seconds = 0

        self.assertEqual(runner.run("Classify."), "neutral")
        self.assertEqual(len(session.calls), 2)

    def test_retries_transient_connection_errors(self):
        session = ExceptionThenResponseSession(
            [requests.ConnectionError("DNS lookup failed")],
            FakeResponse({"choices": [{"message": {"content": "negative"}}]}),
        )

        runner = OpenAICompatiblePromptRunner(
            model="Gemini-2.5-Flash",
            base_url="https://gateway.example/v1",
            api_key="",
            session=session,
        )
        runner.retry_sleep_seconds = 0

        self.assertEqual(runner.run("Classify."), "negative")
        self.assertEqual(len(session.calls), 2)

    def test_retries_empty_length_response_with_larger_token_budget(self):
        session = SequenceSession(
            [
                FakeResponse(
                    {
                        "choices": [
                            {
                                "message": {"content": ""},
                                "finish_reason": "length",
                            }
                        ]
                    }
                ),
                FakeResponse({"choices": [{"message": {"content": "positive"}}]}),
            ]
        )
        env = {
            "OPENAI_COMPAT_MIN_MAX_TOKENS": "128",
            "OPENAI_COMPAT_EMPTY_LENGTH_RETRIES": "1",
        }

        with patch.dict(os.environ, env, clear=False):
            runner = OpenAICompatiblePromptRunner(
                model="Gemini-2.5-Flash",
                base_url="https://gateway.example/v1",
                api_key="",
                session=session,
            )

        self.assertEqual(runner.run("Classify.", max_tokens=16), "positive")
        self.assertEqual(session.calls[0]["json"]["max_tokens"], 128)
        self.assertEqual(session.calls[1]["json"]["max_tokens"], 256)

    def test_retries_empty_stop_response_without_increasing_token_budget(self):
        session = SequenceSession(
            [
                FakeResponse(
                    {
                        "choices": [
                            {
                                "message": {"content": ""},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ),
                FakeResponse({"choices": [{"message": {"content": "neutral"}}]}),
            ]
        )
        env = {
            "OPENAI_COMPAT_MIN_MAX_TOKENS": "128",
            "OPENAI_COMPAT_EMPTY_CONTENT_RETRIES": "1",
        }

        with patch.dict(os.environ, env, clear=False):
            runner = OpenAICompatiblePromptRunner(
                model="Gemini-2.5-Flash",
                base_url="https://gateway.example/v1",
                api_key="",
                session=session,
            )

        self.assertEqual(runner.run("Classify.", max_tokens=16), "neutral")
        self.assertEqual(session.calls[0]["json"]["max_tokens"], 128)
        self.assertEqual(session.calls[1]["json"]["max_tokens"], 128)

    def test_api_key_header_is_optional_for_tokenized_gateway_urls(self):
        session = FakeSession({"choices": [{"message": {"content": "neutral"}}]})

        runner = OpenAICompatiblePromptRunner(
            model="Gemini-2.5-Flash",
            base_url="https://gateway.example/token/v1",
            api_key="",
            session=session,
        )

        self.assertEqual(runner.run("Classify."), "neutral")
        self.assertNotIn("Authorization", session.calls[0]["headers"])

    def test_empty_response_content_raises_clear_error(self):
        session = FakeSession(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": ""},
                    }
                ]
            }
        )
        runner = OpenAICompatiblePromptRunner(
            model="Gemini-2.5-Flash",
            base_url="https://gateway.example/v1",
            api_key="",
            session=session,
        )

        with patch.dict(os.environ, {"OPENAI_COMPAT_EMPTY_CONTENT_RETRIES": "0"}, clear=False):
            with self.assertRaisesRegex(ValueError, "empty content.*finish_reason=stop.*choice="):
                runner.run("Classify.")

    def test_empty_response_writes_diagnostic_file_without_prompt(self):
        session = FakeSession(
            {
                "id": "response-id",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": ""},
                    }
                ],
            }
        )

        diagnostic_path = Path(".test_tmp") / "empty_response.json"
        diagnostic_path.parent.mkdir(exist_ok=True)

        env = {"OPENAI_COMPAT_EMPTY_RESPONSE_PATH": str(diagnostic_path)}
        runner = OpenAICompatiblePromptRunner(
            model="Gemini-2.5-Flash",
            base_url="https://gateway.example/v1",
            api_key="",
            session=session,
        )

        with patch.dict(os.environ, env, clear=False):
            with self.assertRaisesRegex(ValueError, "diagnostic_path=.*empty_response.json"):
                runner.run("Sensitive prompt should not be written.")

        self.assertTrue(diagnostic_path.exists())
        diagnostic_text = diagnostic_path.read_text(encoding="utf-8")
        self.assertIn("response-id", diagnostic_text)
        self.assertIn("finish_reason", diagnostic_text)
        self.assertNotIn("Sensitive prompt", diagnostic_text)

    def test_prompt_runner_selects_openai_compatible_backend_from_env(self):
        env = {
            "PROMPT_BACKEND": "openai_compatible",
            "OPENAI_COMPAT_BASE_URL": "https://gateway.example/v1",
            "OPENAI_COMPAT_MODEL": "Gemini-2.5-Flash",
        }

        with patch.dict(os.environ, env, clear=True):
            runner = PromptRunner()

        self.assertIsInstance(runner.runner, OpenAICompatiblePromptRunner)

    def test_get_model_id_reads_openai_compatible_model_name(self):
        env = {
            "PROMPT_BACKEND": "openai_compatible",
            "OPENAI_COMPAT_MODEL": "Gemini-2.5-Flash",
        }

        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_model_id(), "Gemini-2.5-Flash")


if __name__ == "__main__":
    unittest.main()
