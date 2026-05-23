import re
import time
import asyncio
from datetime import datetime
from langchain_groq import ChatGroq


class RateLimitedChatGroq(ChatGroq):
    """
    Directly inherits from ChatGroq to guarantee all generation 
    entry points (like RAGAS calling .agenerate_prompt()) hit our rate limiter.
    """
    delay_seconds: float = 4.0

    def _extract_wait_time(self, error_msg: str) -> float:
        """Parses Groq's token error message to extract backoff seconds."""
        match = re.search(r"Please try again in (?:(\d+)m)?([\d.]+)s", error_msg)
        print(f"Extracted wait time from error message: {match.groups() if match else 'No match found'}")
        if match:
            minutes = float(match.group(1)) if match.group(1) else 0.0
            seconds = float(match.group(2))
            print(f"Parsed wait time - {(minutes * 60) + seconds + 3}")
            return (minutes * 60) + seconds + 3  # Add a safe extra buffer
        return 60.0

    def _combine_llm_outputs(self, llm_outputs: list) -> dict:
        """
        Patches a structural bug in langchain_groq where nested token usage dictionaries
        (like completion_tokens_details) cause a TypeError during batching (for openai model).
        """
        cleaned_outputs = []
        for output in llm_outputs:
            if output is None:
                continue
            
            # Make a shallow copy to safely scrub the nested dicts
            output_copy = dict(output)
            if "token_usage" in output_copy and isinstance(output_copy["token_usage"], dict):
                usage_copy = dict(output_copy["token_usage"])
                # Remove any nested dictionary fields that break standard arithmetic operators
                keys_to_drop = [k for k, v in usage_copy.items() if isinstance(v, dict)]
                for k in keys_to_drop:
                    del usage_copy[k]
                output_copy["token_usage"] = usage_copy
            
            cleaned_outputs.append(output_copy)
        return super()._combine_llm_outputs(cleaned_outputs)

    def _handle_rate_limit(self, error_str: str) -> float:
        """
        Handles 429 errors.
        Returns wait_time for TPM (worth retrying).
        Raises RuntimeError for TPD (no point retrying).
        """
        if "tokens per day" in error_str.lower():
            raise RuntimeError(
                f"Daily token quota exhausted for {self.model_name}. "
                f"Evaluation cannot proceed today. "
                f"Try again tomorrow or switch judge model."
            )

        wait_time = self._extract_wait_time(error_str)

        if wait_time > 300:
            raise RuntimeError(
                f"Excessive backoff ({wait_time:.0f}s) for "
                f"{self.model_name}. "
                f"Daily quota likely exhausted. Try again tomorrow."
            )

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"TPM limit — backing off {wait_time:.1f}s..."
        )
        return wait_time
    
    async def _agenerate(self, *args, **kwargs):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [Async Judge] Pacing delay: {self.delay_seconds}s...")
        await asyncio.sleep(self.delay_seconds)

        while True:
            try:
                return await super()._agenerate(*args, **kwargs)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit_exceeded" in error_str:
                    wait_time = self._handle_rate_limit(error_str)
                    await asyncio.sleep(wait_time)
                    continue
                raise

    def _generate(self, *args, **kwargs):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [sync Judge] Pacing delay: {self.delay_seconds}s...")
        time.sleep(self.delay_seconds)

        while True:
            try:
                return super()._generate(*args, **kwargs)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit_exceeded" in error_str:
                    wait_time = self._handle_rate_limit(error_str)
                    time.sleep(wait_time)
                    continue
                raise
