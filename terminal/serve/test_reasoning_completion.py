"""Reasoning-only natural stops through the same HTTP stream used by Zed."""
import json
from pathlib import Path
import unittest
import urllib.error
import urllib.request

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


TOOL = {"type": "function", "function": {"name": "read_file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
CALL = "<tool_call>\n<function=read_file>\n<parameter=path>README.md</parameter>\n</function>\n</tool_call>"


class StoppedReasoningEngine(MockEngine):
    def __init__(self, tok):
        super().__init__(tok, "unused", max_context=131072)
        self.parts = ['I checked the result: "ready"</think>', 'The work is complete.']
        self.prompts, self.samplings = [], []
        self.cancel_at = None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        index = len(self.prompts)
        self.prompts.append(list(ids)); self.samplings.append(dict(sampling or {}))
        text = self.parts[min(index, len(self.parts) - 1)]
        tokens = (self.tok.encode(text, parse_special=True) + self.tok.encode('<|im_end|>', parse_special=True))[:max_new]
        generated, stopped = 0, False
        try:
            for token in tokens:
                if cancel.is_set(): return
                generated += 1
                if self.cancel_at == (index, generated): cancel.set()
                stopped = token in self.tok.encode('<|im_end|>', parse_special=True)
                yield token
        finally:
            self.last = dict(prompt_tokens=len(ids), generated=generated,
                             reused=3 if not index else len(ids) - 1,
                             prompt_read=len(ids) - 3 if not index else 1,
                             prompt_ms=10.0, decode_ms=20.0, finish='stop' if stopped else 'length')


class ReasoningCompletion(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer(); self.engine = StoppedReasoningEngine(self.tok)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        self.svc.reasoning_close_retry = True
        self.svc.reasoning_auto_budget_tokens = 4096
        self.httpd = serve(self.svc, port=0)
        self.base = f'http://127.0.0.1:{self.httpd.server_address[1]}'

    def tearDown(self):
        self.httpd.shutdown(); self.httpd.server_close()

    def post(self, path='/v1/chat/completions', **extra):
        body = {'model': 'm', 'max_tokens': 768, 'reasoning_effort': 'high',
                'messages': [{'role': 'user', 'content': 'Finish the work and report the result.'}], **extra}
        if path == '/v1/responses':
            body['input'] = 'Finish the work and report the result.'
            body['max_output_tokens'] = body.pop('max_tokens'); body.pop('messages')
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
        try: response = urllib.request.urlopen(request, timeout=10)
        except urllib.error.HTTPError as error: response = error
        with response:
            raw = response.read().decode()
            return response.status, raw if extra.get('stream') else json.loads(raw)

    def test_closed_thinking_without_an_answer_is_continued(self):
        status, body = self.post()
        self.assertEqual(status, 200, body)
        self.assertEqual(body['choices'][0]['message']['content'], 'The work is complete.')
        self.assertEqual(len(self.engine.prompts), 2)

    def choice(self, **extra):
        status, body = self.post(**extra)
        self.assertEqual(status, 200, body)
        return body['choices'][0], body

    def test_open_thinking_is_closed_once(self):
        self.engine.parts[0] = 'Let me look at the output directory.'
        choice, _ = self.choice()
        self.assertEqual(choice['message']['content'], self.engine.parts[1])
        self.assertTrue(self.tok.decode(self.engine.prompts[1]).endswith('</think>\n\n'))
        self.assertEqual(self.svc.history[-1]['reasoning_answer_retries'], 1)

    def test_closed_and_whitespace_only_content_are_not_answers(self):
        self.engine.parts[0] = 'Checked.</think>\n\n \n'
        choice, _ = self.choice()
        self.assertEqual(choice['message']['content'].strip(), self.engine.parts[1])
        self.assertEqual(self.tok.decode(self.engine.prompts[1]).count('</think>'), 1)

    def test_unicode_reasoning_is_preserved(self):
        self.engine.parts[0] = 'Перевірку завершено. 🙂</think>'
        choice, _ = self.choice()
        self.assertEqual(choice['message']['reasoning_content'], 'Перевірку завершено. 🙂')
        self.assertEqual(choice['message']['content'], self.engine.parts[1])

    def test_disabled_policy_preserves_the_original_single_generation(self):
        self.svc.reasoning_close_retry = False
        choice, _ = self.choice()
        self.assertFalse(choice['message']['content'])
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertNotIn('reasoning_answer_retries', self.svc.history[-1])

    def test_an_existing_answer_is_not_generated_again(self):
        self.engine.parts[0] = 'Checked.</think>\n\nAlready done.'
        choice, _ = self.choice()
        self.assertEqual(choice['message']['content'], 'Already done.')
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(self.svc.history[-1]['reasoning_answer_retries'], 0)

    def test_thinking_off_is_not_retried(self):
        self.engine.parts[0] = 'Done.'
        choice, _ = self.choice(reasoning_effort='none')
        self.assertEqual(choice['message']['content'], 'Done.')
        self.assertEqual(len(self.engine.prompts), 1)

    def test_explicit_stop_sequence_is_respected(self):
        self.engine.parts[0] = 'Checked.</think>\n\nEND'
        choice, _ = self.choice(stop=['END'])
        self.assertFalse(choice['message']['content'])
        self.assertEqual(choice['finish_reason'], 'stop')
        self.assertEqual(len(self.engine.prompts), 1)

    def test_a_real_answer_held_as_a_partial_stop_sequence_is_not_retried(self):
        self.engine.parts[0] = 'Checked.</think>\n\nDone.'
        choice, _ = self.choice(stop=['Done. Later'])
        self.assertEqual(choice['message']['content'], 'Done.')
        self.assertEqual(len(self.engine.prompts), 1)

    def test_a_token_limit_is_not_retried(self):
        self.engine.parts[0] = 'Thinking. ' * 30
        self.svc.answer_room = False; self.svc.reasoning_auto_budget_tokens = 0
        choice, body = self.choice(max_tokens=40)
        self.assertEqual(choice['finish_reason'], 'length')
        self.assertLessEqual(body['usage']['completion_tokens'], 40)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_no_extra_budget_is_created_for_the_closing_marker(self):
        self.engine.parts[0] = 'Checked.'
        self.svc.answer_room = False; self.svc.reasoning_auto_budget_tokens = 0
        cap = len(self.tok.encode('Checked.')) + 2
        choice, body = self.choice(max_tokens=cap)
        self.assertEqual(choice['finish_reason'], 'length')
        self.assertLessEqual(body['usage']['completion_tokens'], cap)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_a_canonical_closed_prefix_needs_no_extra_newlines(self):
        self.engine.parts[0] = 'Checked.</think>\n\n'
        self.choice()
        resumed = self.tok.decode(self.engine.prompts[1])
        self.assertTrue(resumed.endswith('Checked.</think>\n\n'))
        self.assertFalse(resumed.endswith('</think>\n\n\n'))

    def test_cancellation_is_not_retried(self):
        self.engine.cancel_at = (0, 5)
        self.choice()
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(self.svc.history[-1]['finish'], 'cancel')

    def test_two_empty_stops_are_a_server_error_and_leave_the_engine_usable(self):
        self.engine.parts[1] = ''
        status, body = self.post()
        self.assertEqual(status, 502, body)
        self.assertEqual(body['error']['code'], 'reasoning_only_completion')
        self.assertEqual(len(self.engine.prompts), 2)
        self.assertEqual(self.svc.history[-1]['finish'], 'error')
        self.assertFalse(self.svc.status['busy'])
        self.engine.parts.append('Checked.</think>\n\nHealthy.')
        choice, _ = self.choice()
        self.assertEqual(choice['message']['content'], 'Healthy.')
        self.assertEqual(len(self.engine.prompts), 3)

    def test_whitespace_on_the_retry_is_not_success(self):
        self.engine.parts[1] = '\n \n'
        status, body = self.post()
        self.assertEqual(status, 502, body)
        self.assertEqual(len(self.engine.prompts), 2)

    @staticmethod
    def events(raw):
        return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]

    def test_zed_sse_has_one_final_stop_and_one_done_after_the_answer(self):
        status, raw = self.post(stream=True)
        self.assertEqual(status, 200, raw)
        events = self.events(raw)
        choices = [event['choices'][0] for event in events if event.get('choices')]
        text = ''.join(c['delta'].get('content', '') for c in choices)
        self.assertEqual(text, self.engine.parts[1])
        self.assertEqual([c['finish_reason'] for c in choices if c['finish_reason']], ['stop'])
        self.assertEqual(raw.count('data: [DONE]'), 1)
        self.assertEqual(sum(c['delta'].get('role') == 'assistant' for c in choices), 1)
        self.assertEqual(len(self.engine.prompts), 2)

    def test_sse_reports_error_instead_of_empty_success_after_two_stops(self):
        self.engine.parts[1] = ''
        status, raw = self.post(stream=True)
        self.assertEqual(status, 200, raw)
        events = self.events(raw)
        self.assertTrue(any('reasoning-only completion' in e.get('error', {}).get('message', '') for e in events))
        self.assertFalse(any(c['finish_reason'] == 'stop' for e in events for c in e.get('choices', [])))
        self.assertEqual(raw.count('data: [DONE]'), 1)
        self.assertEqual(len(self.engine.prompts), 2)

    def test_pending_calls_inside_reasoning_are_released_once_without_retry(self):
        self.engine.parts[0] = 'I will read the file.\n' + CALL
        choice, _ = self.choice(tools=[TOOL])
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        self.assertEqual(len(choice['message']['tool_calls']), 1)
        self.assertEqual(json.loads(choice['message']['tool_calls'][0]['function']['arguments']), {'path': 'README.md'})
        self.assertEqual(len(self.engine.prompts), 1)

    def test_a_visible_tool_call_is_not_retried(self):
        self.engine.parts[0] = 'Checked.</think>\n\n' + CALL
        choice, _ = self.choice(tools=[TOOL])
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        self.assertEqual(len(choice['message']['tool_calls']), 1)
        self.assertEqual(len(self.engine.prompts), 1)

    def test_forced_tool_opening_is_preserved_on_the_continuation(self):
        self.engine.parts = ['Let me inspect README.', '<parameter=path>README.md</parameter>\n</function>\n</tool_call>']
        choice, _ = self.choice(tools=[TOOL], tool_choice={'type': 'function', 'function': {'name': 'read_file'}})
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        self.assertEqual(json.loads(choice['message']['tool_calls'][0]['function']['arguments']), {'path': 'README.md'})
        self.assertEqual(len(self.engine.prompts), 2)
        self.assertIn('</think>\n\n<tool_call>', self.tok.decode(self.engine.prompts[1]))

    def test_sampling_and_original_input_accounting_are_preserved(self):
        _, body = self.choice(temperature=0, presence_penalty=0, top_p=0.8, top_k=12, seed=42)
        self.assertEqual(self.engine.samplings[0], self.engine.samplings[1])
        self.assertEqual(body['usage']['prompt_tokens'], len(self.engine.prompts[0]))
        self.assertEqual(body['usage']['prompt_tokens_details']['cached_tokens'], 3)
        self.assertLessEqual(body['usage']['completion_tokens'], 768)

    def test_continuation_uses_the_original_cap(self):
        self.engine.parts[1] = 'A long answer. ' * 30
        choice, body = self.choice(max_tokens=100)
        self.assertEqual(choice['finish_reason'], 'length')
        self.assertTrue(choice['message']['content'])
        self.assertEqual(body['usage']['completion_tokens'], 100)
        self.assertEqual(len(self.engine.prompts), 2)

    def test_anthropic_text_follows_the_thinking(self):
        status, body = self.post('/v1/messages', thinking={'type': 'enabled', 'budget_tokens': 4096})
        self.assertEqual(status, 200, body)
        self.assertEqual(''.join(p.get('text', '') for p in body['content']), self.engine.parts[1])
        self.assertEqual(body['stop_reason'], 'end_turn')
        self.assertEqual(len(self.engine.prompts), 2)

    def test_anthropic_stream_finishes_after_text(self):
        status, raw = self.post('/v1/messages', stream=True, thinking={'type': 'enabled', 'budget_tokens': 4096})
        self.assertEqual(status, 200, raw)
        events = self.events(raw)
        self.assertEqual(''.join(e.get('delta', {}).get('text', '') for e in events), self.engine.parts[1])
        self.assertEqual(sum(e['type'] == 'message_stop' for e in events), 1)

    def test_responses_has_a_completed_output_text(self):
        status, body = self.post('/v1/responses', reasoning={'effort': 'high'})
        self.assertEqual(status, 200, body)
        self.assertEqual(body['status'], 'completed')
        self.assertIn(self.engine.parts[1], json.dumps(body))
        self.assertEqual(len(self.engine.prompts), 2)

    def test_responses_stream_has_one_completion_after_text(self):
        status, raw = self.post('/v1/responses', stream=True, reasoning={'effort': 'high'})
        self.assertEqual(status, 200, raw)
        events = self.events(raw)
        self.assertEqual(sum(e['type'] == 'response.completed' for e in events), 1)
        self.assertIn(self.engine.parts[1], raw)

    def test_responses_empty_retry_is_not_a_completed_response(self):
        self.engine.parts[1] = ''
        status, body = self.post('/v1/responses', reasoning={'effort': 'high'})
        self.assertEqual(status, 502, body)
        self.assertEqual(body['error']['code'], 'reasoning_only_completion')

    def test_responses_stream_reports_failure_after_two_empty_stops(self):
        self.engine.parts[1] = ''
        status, raw = self.post('/v1/responses', stream=True, reasoning={'effort': 'high'})
        self.assertEqual(status, 200, raw)
        events = self.events(raw)
        self.assertEqual(sum(e['type'] == 'response.failed' for e in events), 1)
        self.assertFalse(any(e['type'] == 'response.completed' for e in events))

    def test_anthropic_empty_retry_is_a_server_error(self):
        self.engine.parts[1] = ''
        status, body = self.post('/v1/messages', thinking={'type': 'enabled', 'budget_tokens': 4096})
        self.assertEqual(status, 502, body)
        self.assertEqual(body['error']['code'], 'reasoning_only_completion')

    def test_anthropic_stream_reports_an_error_after_two_empty_stops(self):
        self.engine.parts[1] = ''
        status, raw = self.post('/v1/messages', stream=True, thinking={'type': 'enabled', 'budget_tokens': 4096})
        self.assertEqual(status, 200, raw)
        events = self.events(raw)
        self.assertEqual(sum(e['type'] == 'error' for e in events), 1)
        self.assertFalse(any(e['type'] == 'message_stop' for e in events))

    def test_recovery_and_failure_are_recorded_in_the_current_engine_log(self):
        import tempfile
        self.engine.parts[1] = ''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'engine.log'; self.engine.log_path = str(path)
            status, _ = self.post()
            self.assertEqual(status, 502)
            note = path.read_text()
            self.assertEqual(note.count('continuing once'), 1)
            self.assertEqual(note.count('returning an error'), 1)
            self.assertNotIn('I checked the result', note)


if __name__ == '__main__': unittest.main()
