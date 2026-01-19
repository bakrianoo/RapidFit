"""Chat completion client."""

from openai import OpenAI


class ChatClient:
    """Chat completion client using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model_id: str = "gpt-4.1-mini",
    ) -> None:
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key.
            base_url: Optional custom base URL for API.
            model_id: Model identifier to use.
        """
        self.model_id = model_id
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.call_count = 0
        self.total_tokens = 0

    def complete(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        """
        Get completion from the model.

        Args:
            prompt: User prompt.
            system: Optional system message.
            temperature: Sampling temperature.

        Returns:
            Model response text.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
        )
        self.call_count += 1
        if response.usage:
            self.total_tokens += response.usage.total_tokens
        return response.choices[0].message.content or ""
