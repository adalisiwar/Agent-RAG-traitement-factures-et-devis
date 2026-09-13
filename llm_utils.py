import time
from google.genai import errors as genai_errors


def is_daily_quota_exhausted(api_error):
    message = str(api_error)
    return "PerDay" in message or "per day" in message.lower()


def call_with_retry(fn, max_retries=4, base_backoff=5, on_quota_exhausted=None):
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except genai_errors.ClientError as api_error:
            if api_error.code == 429:
                if is_daily_quota_exhausted(api_error):
                    if on_quota_exhausted:
                        on_quota_exhausted(api_error)
                    raise
                backoff = base_backoff * attempt
                time.sleep(backoff)
                continue
            raise
        except genai_errors.ServerError:
            backoff = base_backoff * attempt
            time.sleep(backoff)

    raise RuntimeError(f"Gemini API call failed after {max_retries} attempts.")
