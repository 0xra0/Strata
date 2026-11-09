#!/usr/bin/env python3
"""Two short, isolated translation calls; record HTTP status, empty bodies and timings."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request


def fetch(url, body=None, timeout=60):
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('STRATA_API_KEY'): headers['Authorization'] = 'Bearer ' + os.environ['STRATA_API_KEY']
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode('utf-8') if body is not None else None,
                                 headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        with e: status, raw = e.code, e.read()
    except (OSError, urllib.error.URLError) as e:
        return {'http_status': None, 'error': str(e), 'elapsed_s': round(time.perf_counter() - started, 3)}
    result = {'http_status': status, 'body_bytes': len(raw), 'elapsed_s': round(time.perf_counter() - started, 3)}
    if not raw:
        result['error'] = 'Empty HTTP response body'
    else:
        try: result['response'] = json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            result.update(error='Response was not valid JSON', body_preview=raw[:4096].decode('utf-8', errors='replace'))
    return result


def check(base, model=None):
    base = base.rstrip('/')
    report = {'models': fetch(base + '/v1/models'), 'requests': []}
    if model is None:
        models = report['models'].get('response', {}).get('data') or []
        if not models:
            report['error'] = 'No model returned; start Strata and wait for ready before running this check'
            return report
        model = models[0]['id']
    body = {'model': model, 'messages': [{'role': 'user', 'content':
            'Translate Ukrainian to English. Output only the translation:\nПривіт! Як у тебе справи?'}],
            'reasoning_effort': 'none', 'temperature': 0, 'presence_penalty': 0, 'max_tokens': 128}
    for label in ('first', 'repeat'):
        result = fetch(base + '/v1/chat/completions', body)
        result['request_label'] = label
        report['requests'].append(result)
        if result.get('http_status') != 200 or result.get('error'): break
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='http://127.0.0.1:8099')
    p.add_argument('--model')
    args = p.parse_args()
    print(json.dumps(check(args.base, args.model), ensure_ascii=False, indent=2))
