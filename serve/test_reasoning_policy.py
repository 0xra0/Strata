import unittest
from serve.reasoning_policy import auto_reasoning_budget


class Policy(unittest.TestCase):
    def test_unlimited_context_does_not_scale_the_allowance(self):
        for cap in (8192, 32000, 68931, 131072):
            self.assertEqual(auto_reasoning_budget(4096, {"max_tokens": cap}), 4096)

    def test_the_web_apps_default_high_is_bounded(self):
        self.assertEqual(auto_reasoning_budget(4096, {"reasoning_effort": "high"}), 4096)

    def test_effort_levels(self):
        self.assertEqual(auto_reasoning_budget(4096, {"reasoning_effort": "low"}), 2048)
        self.assertEqual(auto_reasoning_budget(4096, {"reasoning": {"effort": "xhigh"}}), 8192)
        self.assertEqual(auto_reasoning_budget(4096, {}, effort="xhigh"), 8192)
        self.assertIsNone(auto_reasoning_budget(4096, {"reasoning_effort": "none"}))

    def test_explicit_request_shared_and_config_budgets_win(self):
        for budget in (0, -1, 20, 16000):
            self.assertIsNone(auto_reasoning_budget(4096, {"reasoning_budget_tokens": budget}))
        self.assertIsNone(auto_reasoning_budget(4096, {}, configured_budget=10000))
        self.assertIsNone(auto_reasoning_budget(0, {}))

    def test_invalid_configuration(self):
        for value in (True, -1, "4096", 4.5, None):
            with self.assertRaises(ValueError): auto_reasoning_budget(value, {})


class Integration(unittest.TestCase):
    def setUp(self):
        from serve.test_server import ThinkingBudget
        ThinkingBudget.setUp(self)
        self.svc.reasoning_auto_budget_tokens = 20

    def tearDown(self):
        from serve.test_server import ThinkingBudget
        ThinkingBudget.tearDown(self)

    def post(self, path, body):
        from serve.test_server import ThinkingBudget
        return ThinkingBudget.post(self, path, body)

    def chat(self, **extra):
        return self.post('/v1/chat/completions', {"model": "m", "messages": [{"role": "user", "content": "2+2?"}],
                                               "max_tokens": 400, **extra})

    def test_auto_budget_continues_to_answer(self):
        from serve.server import REASONING_WRAP_UP
        status, body = self.chat(reasoning_effort="high")
        self.assertEqual(status, 200, body)
        message = body['choices'][0]['message']
        self.assertEqual(message['reasoning_content'], self.engine.THOUGHT[:20] + REASONING_WRAP_UP.split('</think>')[0])
        self.assertEqual(message['content'], self.engine.ANSWER)
        self.assertEqual(len(self.engine.prompts), 2)
        self.assertLessEqual(body['usage']['completion_tokens'], 400)

    def test_streaming_continues_to_answer(self):
        status, body = self.chat(stream=True)
        self.assertEqual(status, 200, body)
        self.assertIn('[DONE]', body)
        import json
        chunks = [json.loads(s[6:]) for s in body.splitlines() if s.startswith('data: {')]
        text = ''.join(c['choices'][0].get('delta', {}).get('content', '') for c in chunks if c.get('choices'))
        self.assertEqual(text, self.engine.ANSWER)
        self.assertEqual(len(self.engine.prompts), 2)

    def test_responses_api_uses_the_same_allowance(self):
        status, body = self.post('/v1/responses', {"model": "m", "input": "2+2?", "max_output_tokens": 400})
        self.assertEqual(status, 200, body)
        import json
        self.assertIn(self.engine.ANSWER, json.dumps(body))
        self.assertEqual(len(self.engine.prompts), 2)

    def test_an_explicit_budget_is_not_replaced(self):
        status, body = self.chat(reasoning_budget_tokens=10000)
        self.assertEqual(status, 200, body)
        self.assertEqual(body['choices'][0]['message']['reasoning_content'], self.engine.THOUGHT)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_an_explicit_zero_disables_auto(self):
        status, body = self.chat(reasoning_budget_tokens=0)
        self.assertEqual(status, 200, body)
        self.assertEqual(body['choices'][0]['message']['reasoning_content'], self.engine.THOUGHT)

    def test_anthropic_continuation(self):
        status, body = self.post('/v1/messages', {"model": "m", "max_tokens": 400,
            "messages": [{"role": "user", "content": "2+2?"}]})
        self.assertEqual(status, 200, body)
        self.assertTrue(any(b.get('text') == self.engine.ANSWER for b in body['content']))


if __name__ == '__main__': unittest.main()
